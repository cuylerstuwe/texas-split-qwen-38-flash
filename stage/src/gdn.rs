//! Qwen's GDN attention sub-block: real affine8 weights, staged BF16 semantic
//! boundaries, FP32 decay/state, sigmoid output gate. No HC, MoE, QSA or model.
use crate::{dense::Affine8, projection::Pack};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::json;
pub const H: usize = 2560;
pub const DK: usize = 128;
pub const DV: usize = 128;
pub const HK: usize = 16;
pub const HV: usize = 48;
pub const C: usize = 2 * HK * DK + HV * DV;
pub const V: usize = HV * DV;
pub const STATE: usize = HV * DV * DK;
pub const MAX_T: usize = 16;
fn rounded(x: f64) -> f64 {
    crate::rounding::bf16_from_f64(x).to_f64()
}
fn number(x: u16) -> f64 {
    bf16::from_bits(x).to_f64()
}
/// Staged BF16 sigmoid matching the source expression, evaluated with an
/// independent scalar exp before each explicit BF16 arithmetic boundary.
fn sigmoid(x: f64) -> f64 {
    let small = rounded(1. / rounded(1. + rounded(x.abs().exp())));
    if x < 0. { small } else { rounded(1. - small) }
}
pub struct Weights {
    pub projections: Vec<Affine8>, // qkv,z,a,b,out
    pub conv: Vec<u16>,
    pub a_log: Vec<u16>,
    pub dt_bias: Vec<u16>,
    pub norm: Vec<u16>,
    pub provenance: serde_json::Value,
}
impl Weights {
    pub fn load(pack: &Pack, layer: usize) -> Result<Self> {
        ensure!(layer < 48 && layer % 4 != 3, "not a bounded GDN layer");
        let p = format!("language_model.model.layers.{layer}.linear_attn");
        let mut projections = vec![];
        for (suffix, k, n) in [
            ("in_proj_qkv", H, C),
            ("in_proj_z", H, V),
            ("in_proj_a", H, HV),
            ("in_proj_b", H, HV),
            ("out_proj", V, H),
        ] {
            let w = Affine8::load(pack, &format!("{p}.{suffix}"), k)?;
            ensure!(w.n == n, "GDN projection geometry");
            projections.push(w);
        }
        let mut provenance = vec![];
        let mut small = |suffix: &str, shape: &[u64]| -> Result<Vec<u16>> {
            let (t, b, prov) = pack.tensor(&format!("{p}.{suffix}"))?;
            ensure!(
                t.dtype == "BF16" && t.shape == shape,
                "GDN parameter shape/dtype"
            );
            provenance.push(prov);
            let v: Vec<_> = b
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            ensure!(
                v.iter().all(|&x| bf16::from_bits(x).is_finite()),
                "nonfinite GDN weight"
            );
            Ok(v)
        };
        let conv = small("conv1d.weight", &[C as u64, 4, 1])?;
        let a_log = small("A_log", &[HV as u64])?;
        let dt_bias = small("dt_bias", &[HV as u64])?;
        let norm = small("norm.weight", &[DV as u64])?;
        Ok(Self {
            provenance: json!({"layer":layer,"projections":projections.iter().map(|w|&w.provenance).collect::<Vec<_>>(),"parameters":provenance}),
            projections,
            conv,
            a_log,
            dt_bias,
            norm,
        })
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.projections.len() == 5
                && self.conv.len() == C * 4
                && self.a_log.len() == HV
                && self.dt_bias.len() == HV
                && self.norm.len() == DV,
            "GDN weights"
        );
        for (w, (k, n)) in self
            .projections
            .iter()
            .zip([(H, C), (H, V), (H, HV), (H, HV), (V, H)])
        {
            w.validate()?;
            ensure!((w.k, w.n) == (k, n), "GDN projection shape");
        }
        Ok(())
    }
}
#[derive(Clone)]
pub struct Initial {
    pub conv: Vec<u16>,
    pub state: Vec<f32>,
}
impl Initial {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.conv.len() == 3 * C && self.state.len() == STATE,
            "GDN initial state shape"
        );
        ensure!(
            self.conv.iter().all(|&x| bf16::from_bits(x).is_finite())
                && self.state.iter().all(|x| x.is_finite()),
            "GDN initial state finite"
        );
        Ok(())
    }
}
#[derive(Clone)]
pub struct Core {
    pub y: Vec<u16>,
    pub gated: Vec<u16>,
    pub states: Vec<f32>,
    pub conv: Vec<u16>,
}
/// Independent scalar CPU reference. Dot products/normalization use FP64;
/// each semantic BF16 tensor and each FP32 recurrent checkpoint is rounded.
pub fn reference(w: &Weights, initial: &Initial, projected: &[Vec<u16>]) -> Result<Core> {
    reference_traced(w, initial, projected, |_, _, _, _, _, _, _| {})
}
/// CPU-only observation at the existing prepared-operand boundary. A callback
/// cannot mutate operands or state. Device kernels and recurrence are unchanged.
pub fn reference_traced(
    w: &Weights,
    initial: &Initial,
    projected: &[Vec<u16>],
    trace: impl FnMut(usize, usize, &[f64], &[f64], &[f64], f32, f64),
) -> Result<Core> {
    w.validate()?;
    reference_parameters(
        Parameters {
            conv: &w.conv,
            a_log: &w.a_log,
            dt_bias: &w.dt_bias,
            norm: &w.norm,
        },
        initial,
        projected,
        trace,
    )
}
/// Projection-independent parameters let the same mathematical oracle check
/// both original affine8 and the separately encoded affine4 trunk. GPU code
/// does not call this reference or materialize its dense weights.
#[derive(Clone, Copy)]
pub struct Parameters<'a> {
    pub conv: &'a [u16],
    pub a_log: &'a [u16],
    pub dt_bias: &'a [u16],
    pub norm: &'a [u16],
}
impl Parameters<'_> {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.conv.len() == C * 4
                && self.a_log.len() == HV
                && self.dt_bias.len() == HV
                && self.norm.len() == DV,
            "GDN parameter geometry"
        );
        ensure!(
            self.conv
                .iter()
                .chain(self.a_log)
                .chain(self.dt_bias)
                .chain(self.norm)
                .all(|&v| bf16::from_bits(v).is_finite()),
            "GDN parameter finite"
        );
        Ok(())
    }
}
pub fn reference_parameters(
    w: Parameters<'_>,
    initial: &Initial,
    projected: &[Vec<u16>],
    mut trace: impl FnMut(usize, usize, &[f64], &[f64], &[f64], f32, f64),
) -> Result<Core> {
    w.validate()?;
    initial.validate()?;
    ensure!(
        projected.len() == 4 && !projected[0].is_empty() && projected[0].len() % C == 0,
        "GDN projected inputs"
    );
    let t = projected[0].len() / C;
    ensure!((1..=MAX_T).contains(&t) && projected.iter().zip([C,V,HV,HV]).all(|(x,n)|x.len()==t*n && x.iter().all(|&v|bf16::from_bits(v).is_finite())),"GDN projected geometry");
    let mut combined = initial.conv.clone();
    combined.extend(&projected[0]);
    let mut y = vec![0; t * V];
    let mut gated = vec![0; t * V];
    let mut states = Vec::with_capacity(t * STATE);
    let mut state = initial.state.clone();
    for pos in 0..t {
        let mut act = vec![0f64; C];
        for ch in 0..C {
            let dot = (0..4)
                .map(|tap| number(combined[(pos + tap) * C + ch]) * number(w.conv[ch * 4 + tap]))
                .sum::<f64>();
            let conv = rounded(dot);
            act[ch] = rounded(conv * sigmoid(conv));
        }
        for hk in 0..HK {
            for (off, scale) in [
                (hk * DK, 1. / DK as f64),
                (HK * DK + hk * DK, rounded(1. / (DK as f64).sqrt())),
            ] {
                let inv = 1.
                    / (act[off..off + DK].iter().map(|x| x * x).sum::<f64>() / DK as f64 + 1e-6)
                        .sqrt();
                for x in &mut act[off..off + DK] {
                    *x = rounded(rounded(*x * inv) * scale);
                }
            }
        }
        for hv in 0..HV {
            let hk = hv / (HV / HK);
            let q = &act[hk * DK..(hk + 1) * DK];
            let k = &act[HK * DK + hk * DK..HK * DK + (hk + 1) * DK];
            let apd = (number(projected[2][pos * HV + hv]) + number(w.dt_bias[hv])) as f32;
            let decay = (-number(w.a_log[hv]).exp() * (f64::from(apd).exp().ln_1p())).exp() as f32;
            let beta = sigmoid(number(projected[3][pos * HV + hv]));
            trace(
                pos,
                hv,
                q,
                k,
                &act[2 * HK * DK + hv * DV..2 * HK * DK + (hv + 1) * DV],
                decay,
                beta,
            );
            for dv in 0..DV {
                let row = &mut state[(hv * DV + dv) * DK..(hv * DV + dv + 1) * DK];
                let memory = row
                    .iter()
                    .zip(k)
                    .map(|(&s, &k)| f64::from(s) * f64::from(decay) * k)
                    .sum::<f64>();
                let delta = beta * (act[2 * HK * DK + hv * DV + dv] - memory);
                for (s, &k) in row.iter_mut().zip(k) {
                    *s = (f64::from(*s) * f64::from(decay) + delta * k) as f32;
                }
                let out = row
                    .iter()
                    .zip(q)
                    .map(|(&s, &q)| f64::from(s) * q)
                    .sum::<f64>();
                y[(pos * HV + hv) * DV + dv] = crate::rounding::bf16_from_f64(out).to_bits();
            }
            let base = (pos * HV + hv) * DV;
            let inv = 1.
                / (y[base..base + DV]
                    .iter()
                    .map(|&v| number(v).powi(2))
                    .sum::<f64>()
                    / DV as f64
                    + 1e-6)
                    .sqrt();
            for dv in 0..DV {
                let normed = rounded(number(w.norm[dv]) * rounded(number(y[base + dv]) * inv));
                gated[base + dv] = crate::rounding::bf16_from_f64(
                    normed * sigmoid(number(projected[1][base + dv])),
                )
                .to_bits();
            }
        }
        states.extend(&state);
    }
    Ok(Core {
        y,
        gated,
        states,
        conv: combined[t * C..(t + 3) * C].to_vec(),
    })
}
/// Exact state selection for this sub-block. Token publication and other layer
/// state remain the caller's responsibility. No discarded suffix is replayed.
pub fn rollback(
    initial: &Initial,
    qkv: &[u16],
    states: &[f32],
    accepted: usize,
) -> Result<Initial> {
    initial.validate()?;
    ensure!(
        qkv.len() % C == 0
            && qkv.len() / C <= MAX_T
            && states.len() == qkv.len() / C * STATE
            && accepted <= qkv.len() / C,
        "GDN rollback geometry"
    );
    let mut conv = initial.conv.clone();
    conv.extend_from_slice(&qkv[..accepted * C]);
    let state = if accepted == 0 {
        initial.state.clone()
    } else {
        states[(accepted - 1) * STATE..accepted * STATE].to_vec()
    };
    Ok(Initial {
        conv: conv[accepted * C..(accepted + 3) * C].to_vec(),
        state,
    })
}
