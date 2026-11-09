//! Qwen gated attention and affine8 KV semantics; independent CPU oracle.
//! Inference arithmetic belongs on accelerators. This module's scalar loops
//! are validation/fixture construction, not a CPU token-generation backend.
use crate::{dense::Affine8, projection::Pack, qsa};
use anyhow::{Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::json;
pub const H: usize = 2560;
pub const D: usize = 256;
pub const QH: usize = 24;
pub const KH: usize = 2;
pub const Q: usize = QH * D;
pub const KV: usize = KH * D;
pub const QG: usize = Q * 2;
pub const GROUP: usize = 64;
pub const META: usize = KV / GROUP;
pub const MAX_ROWS: usize = 16;
pub const MAX_REFERENCE_CTX: usize = 32768;
pub fn bits(v: f64) -> u16 {
    crate::rounding::bf16_from_f64(v).to_bits()
}
pub fn num(v: u16) -> f64 {
    bf16::from_bits(v).to_f64()
}
fn round(v: f64) -> f64 {
    num(bits(v))
}
fn finite(v: &[u16]) -> bool {
    v.iter().all(|&v| bf16::from_bits(v).is_finite())
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Packed {
    pub codes: Vec<u8>,
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
}
impl Packed {
    pub fn empty() -> Self {
        Self {
            codes: vec![],
            scales: vec![],
            biases: vec![],
        }
    }
    pub fn validate(&self, tokens: usize) -> Result<()> {
        ensure!(
            tokens <= qsa::MAX_POSITION
                && self.codes.len() == tokens * KV
                && self.scales.len() == tokens * META
                && self.biases.len() == tokens * META,
            "KV packed geometry"
        );
        ensure!(
            finite(&self.scales)
                && finite(&self.biases)
                && self.scales.iter().all(|&v| num(v) != 0.),
            "KV finite nonzero metadata"
        );
        Ok(())
    }
    /// Source affine quantize: FP32 code decision BEFORE rounding metadata.
    /// Positive max starts at0 even for all-negative groups, matching MLX.
    pub fn quantize(v: &[u16]) -> Result<Self> {
        ensure!(
            v.len() % KV == 0 && v.len() / KV <= MAX_REFERENCE_CTX + MAX_ROWS && finite(v),
            "bounded KV quantizer input"
        );
        let mut out = Self::empty();
        for group in v.chunks_exact(GROUP) {
            let values: Vec<f32> = group.iter().map(|&v| bf16::from_bits(v).to_f32()).collect();
            let lo = values.iter().copied().fold(f32::MAX, f32::min);
            let hi = values.iter().copied().fold(0f32, f32::max);
            let mut scale = ((hi - lo) / 255.).max(1e-7);
            let negative_edge = lo.abs() > hi.abs();
            if !negative_edge {
                scale = -scale;
            }
            let edge = if negative_edge { lo } else { hi };
            let q0 = (edge / scale).round();
            let bias = if q0 == 0. {
                0.
            } else {
                scale = edge / q0;
                edge
            };
            ensure!(
                scale.is_finite() && scale != 0. && bias.is_finite(),
                "KV quantizer overflow"
            );
            out.scales.push(bf16::from_f32(scale).to_bits());
            out.biases.push(bf16::from_f32(bias).to_bits());
            for x in values {
                let q = ((x - bias) / scale).round().min(255.);
                ensure!(
                    q.is_finite() && (0.0..=255.0).contains(&q),
                    "KV quantizer code range"
                );
                out.codes.push(q as u8);
            }
        }
        out.validate(v.len() / KV)?;
        Ok(out)
    }
    /// Packed attention's FP32 affine -> BF16 store, not a separate BF16
    /// multiply/add. BF16*integer product is exactly representable in FP32.
    pub fn value(&self, i: usize) -> u16 {
        let g = i / GROUP;
        let x = bf16::from_bits(self.scales[g]).to_f32() * self.codes[i] as f32
            + bf16::from_bits(self.biases[g]).to_f32();
        bf16::from_f32(x).to_bits()
    }
    pub fn prefix(&self, tokens: usize) -> Result<Self> {
        ensure!(tokens <= self.codes.len() / KV, "KV prefix bounds");
        let p = Self {
            codes: self.codes[..tokens * KV].to_vec(),
            scales: self.scales[..tokens * META].to_vec(),
            biases: self.biases[..tokens * META].to_vec(),
        };
        p.validate(tokens)?;
        Ok(p)
    }
    fn extend(&mut self, other: &Self) {
        self.codes.extend(&other.codes);
        self.scales.extend(&other.scales);
        self.biases.extend(&other.biases);
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct State {
    pub base: usize,
    pub len: usize,
    pub k: Packed,
    pub v: Packed,
}
impl State {
    pub fn empty(base: usize) -> Result<Self> {
        let s = Self {
            base,
            len: 0,
            k: Packed::empty(),
            v: Packed::empty(),
        };
        s.validate()?;
        Ok(s)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.base <= qsa::MAX_POSITION && self.len <= qsa::MAX_POSITION - self.base,
            "KV state positions"
        );
        self.k.validate(self.len)?;
        self.v.validate(self.len)?;
        Ok(())
    }
    pub fn accepted(&self, old: usize, n: usize) -> Result<Self> {
        self.validate()?;
        ensure!(old <= self.len && n <= self.len - old, "KV commit bounds");
        let len = old + n;
        Ok(Self {
            base: self.base,
            len,
            k: self.k.prefix(len)?,
            v: self.v.prefix(len)?,
        })
    }
}
pub struct Norms {
    pub q: Vec<u16>,
    pub k: Vec<u16>,
    pub provenance: serde_json::Value,
}
impl Norms {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.q.len() == D && self.k.len() == D && finite(&self.q) && finite(&self.k),
            "attention norm geometry"
        );
        Ok(())
    }
    pub fn load(pack: &Pack, layer: usize) -> Result<Self> {
        ensure!(layer < 48 && layer % 4 == 3, "full attention layer");
        let mut records = vec![];
        let mut get = |kind: &str| -> Result<Vec<u16>> {
            let (t, b, p) = pack.tensor(&format!(
                "language_model.model.layers.{layer}.self_attn.{kind}_norm.weight"
            ))?;
            ensure!(t.dtype == "BF16" && t.shape == [D as u64], "QK norm shape");
            records.push(p);
            Ok(b.chunks_exact(2)
                .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
                .collect())
        };
        let q = get("q")?;
        let k = get("k")?;
        let n = Self {
            q,
            k,
            provenance: json!(records),
        };
        n.validate()?;
        Ok(n)
    }
}
pub struct Weights {
    pub q: Affine8,
    pub k: Affine8,
    pub v: Affine8,
    pub o: Affine8,
    pub norms: Norms,
    pub layer: usize,
}
impl Weights {
    pub fn load(pack: &Pack, layer: usize) -> Result<Self> {
        let norms = Norms::load(pack, layer)?;
        let p = format!("language_model.model.layers.{layer}.self_attn");
        let q = Affine8::load(pack, &format!("{p}.q_proj"), H)?;
        let k = Affine8::load(pack, &format!("{p}.k_proj"), H)?;
        let v = Affine8::load(pack, &format!("{p}.v_proj"), H)?;
        let o = Affine8::load(pack, &format!("{p}.o_proj"), Q)?;
        ensure!(
            q.n == QG && k.n == KV && v.n == KV && o.n == H,
            "target attention projections"
        );
        Ok(Self {
            q,
            k,
            v,
            o,
            norms,
            layer,
        })
    }
    pub fn provenance(&self) -> serde_json::Value {
        json!({"layer":self.layer,"q":self.q.provenance,"k":self.k.provenance,"v":self.v.provenance,"o":self.o.provenance,"norms":self.norms.provenance})
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Input {
    pub qg: Vec<u16>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
}
impl Input {
    pub fn rows(&self) -> usize {
        self.qg.len() / QG
    }
    pub fn validate(&self) -> Result<()> {
        let r = self.rows();
        ensure!(
            (1..=MAX_ROWS).contains(&r)
                && self.qg.len() == r * QG
                && self.k.len() == r * KV
                && self.v.len() == r * KV
                && finite(&self.qg)
                && finite(&self.k)
                && finite(&self.v),
            "attention input shape/finite"
        );
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Prepared {
    pub q: Vec<u16>,
    pub k: Vec<u16>,
    pub gate: Vec<u16>,
}
fn norm_rope(x: &[u16], w: &[u16], pos: usize) -> Vec<u16> {
    let inv = (x.iter().map(|&v| num(v).powi(2)).sum::<f64>() / D as f64 + 1e-6)
        .sqrt()
        .recip();
    let mut n: Vec<_> = x
        .iter()
        .zip(w)
        .map(|(&x, &w)| bits(round(num(x) * inv) * num(w)))
        .collect();
    for (j, f) in qsa::frequencies().into_iter().enumerate() {
        let angle = pos as f32 * f;
        let c = angle.cos() as f64;
        let s = angle.sin() as f64;
        let a = num(n[j]);
        let b = num(n[j + 32]);
        n[j] = bits(a * c - b * s);
        n[j + 32] = bits(b * c + a * s);
    }
    n
}
pub fn prepare(input: &Input, norms: &Norms, start: usize) -> Result<Prepared> {
    input.validate()?;
    norms.validate()?;
    ensure!(
        start
            .checked_add(input.rows())
            .is_some_and(|end| end <= qsa::MAX_POSITION),
        "attention position overflow"
    );
    let mut q = vec![];
    let mut k = vec![];
    let mut gate = vec![];
    for r in 0..input.rows() {
        for h in 0..QH {
            let at = (r * QH + h) * 2 * D;
            q.extend(norm_rope(&input.qg[at..at + D], &norms.q, start + r));
            gate.extend_from_slice(&input.qg[at + D..at + 2 * D]);
        }
        for h in 0..KH {
            let at = (r * KH + h) * D;
            k.extend(norm_rope(&input.k[at..at + D], &norms.k, start + r));
        }
    }
    ensure!(finite(&q) && finite(&k), "attention normalized finite");
    Ok(Prepared { q, k, gate })
}
pub fn append(initial: &State, prepared: &Prepared, v: &[u16]) -> Result<State> {
    initial.validate()?;
    ensure!(
        !prepared.k.is_empty()
            && prepared.k.len() % KV == 0
            && prepared.k.len() / KV <= MAX_ROWS
            && v.len() == prepared.k.len(),
        "KV append shape"
    );
    let len = initial.len + v.len() / KV;
    ensure!(
        len <= qsa::MAX_POSITION - initial.base,
        "KV appended positions"
    );
    let mut k = initial.k.clone();
    let mut vals = initial.v.clone();
    k.extend(&Packed::quantize(&prepared.k)?);
    vals.extend(&Packed::quantize(v)?);
    let state = State {
        base: initial.base,
        len,
        k,
        v: vals,
    };
    state.validate()?;
    Ok(state)
}
/// Stable two-pass FP64 softmax. Selection contains only complete visible
/// blocks; its own causal tail is expanded by qsa::Selection::tokens.
pub fn attend(q: &[u16], state: &State, selection: Option<&qsa::Selection>) -> Result<Vec<u16>> {
    state.validate()?;
    ensure!(
        !q.is_empty()
            && q.len() % Q == 0
            && q.len() / Q <= MAX_ROWS
            && q.len() / Q <= state.len
            && state.len <= MAX_REFERENCE_CTX + MAX_ROWS
            && finite(q),
        "bounded attention oracle"
    );
    let rows = q.len() / Q;
    let offset = state.len - rows;
    if let Some(s) = selection {
        s.validate()?;
        ensure!(
            s.offset == offset && s.rows == rows,
            "attention selection alignment"
        );
    }
    let keys: Vec<Vec<usize>> = (0..rows)
        .map(|r| {
            if let Some(s) = selection {
                s.tokens(r)
            } else {
                Ok((0..=offset + r).collect())
            }
        })
        .collect::<Result<_>>()?;
    let mut out = vec![0; rows * Q];
    std::thread::scope(|scope| {
        let per = (rows * QH).div_ceil(4);
        for (chunk, dst) in out.chunks_mut(per * D).enumerate() {
            let keys = &keys;
            scope.spawn(move || {
                for (hidx, yy) in dst.chunks_mut(D).enumerate() {
                    let i = chunk * per + hidx;
                    let row = i / QH;
                    let h = (i % QH) / (QH / KH);
                    let query = &q[i * D..(i + 1) * D];
                    let mut scores = vec![0.; keys[row].len()];
                    for (s, &key) in scores.iter_mut().zip(&keys[row]) {
                        let at = (key * KH + h) * D;
                        *s = (0..D)
                            .map(|d| num(query[d]) * num(state.k.value(at + d)))
                            .sum::<f64>()
                            / 16.;
                    }
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut z = 0.;
                    for s in &mut scores {
                        *s = (*s - max).exp();
                        z += *s;
                    }
                    let mut acc = [0f64; D];
                    for (&p, &key) in scores.iter().zip(&keys[row]) {
                        let at = (key * KH + h) * D;
                        for d in 0..D {
                            acc[d] += p * num(state.v.value(at + d));
                        }
                    }
                    for d in 0..D {
                        yy[d] = bits(acc[d] / z);
                    }
                }
            });
        }
    });
    ensure!(finite(&out), "attention oracle finite output");
    Ok(out)
}
pub fn gate(attention: &[u16], gate: &[u16]) -> Result<Vec<u16>> {
    ensure!(
        attention.len() == gate.len() && finite(attention) && finite(gate),
        "attention gate geometry"
    );
    Ok(attention
        .iter()
        .zip(gate)
        .map(|(&a, &g)| {
            let v = num(g);
            let small = round(1. / round(1. + round(v.abs().exp())));
            let sigmoid = if v < 0. { small } else { round(1. - small) };
            bits(num(a) * sigmoid)
        })
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kv_constant_signed_edges_and_prefix() {
        let values: Vec<_> = [0., 1., -2., 0.125, -0.125, 17., -17., 1e-8]
            .into_iter()
            .flat_map(|v| vec![bits(v); GROUP])
            .collect();
        let p = Packed::quantize(&values).unwrap();
        assert_eq!(p.codes.len(), KV);
        for i in 0..KV {
            // Below the source's 1e-7 quantizer floor, q0 is zero and the
            // tiny constant group intentionally rounds to zero.
            assert_eq!(p.value(i), if i < 7 * GROUP { values[i] } else { 0 });
        }
        assert!(num(p.scales[1]) < 0.);
        assert!(num(p.scales[2]) > 0.);
        assert_eq!(p.prefix(0).unwrap(), Packed::empty());
        assert!(p.prefix(2).is_err());
        let mut bad = values;
        bad[0] = bf16::NAN.to_bits();
        assert!(Packed::quantize(&bad).is_err());
    }
    #[test]
    fn head_interleaved_gate_and_partial_rope() {
        let mut x = Input {
            qg: vec![0; QG],
            k: vec![0; KV],
            v: vec![0; KV],
        };
        for h in 0..QH {
            x.qg[h * 2 * D + 200] = bits((h + 1) as f64);
            x.qg[h * 2 * D + D + 201] = bits(-(h as f64));
        }
        let n = Norms {
            q: vec![bf16::ONE.to_bits(); D],
            k: vec![bf16::ONE.to_bits(); D],
            provenance: json!(null),
        };
        let a = prepare(&x, &n, 0).unwrap();
        let b = prepare(&x, &n, 262143).unwrap();
        assert!(a.q.iter().zip(&b.q).all(|(&x, &y)| num(x) == num(y)));
        for h in 0..QH {
            assert_eq!(a.gate[h * D + 201], bits(-(h as f64)));
            assert_eq!(a.q[h * D], 0);
            assert!(num(a.q[h * D + 200]) > 15.);
        }
        assert!(prepare(&x, &n, usize::MAX).is_err());
    }
    #[test]
    fn causal_uniform_attention_and_kv_commit() {
        let mut s = State::empty(17).unwrap();
        let vals: Vec<_> = [1., 2., 4., 8.]
            .into_iter()
            .flat_map(|v| vec![bits(v); KV])
            .collect();
        s.len = 4;
        s.k = Packed::quantize(&vec![0; 4 * KV]).unwrap();
        s.v = Packed::quantize(&vals).unwrap();
        let y = attend(&vec![0; 2 * Q], &s, None).unwrap();
        assert!(y[..Q].iter().all(|&v| v == bits(7. / 3.)));
        assert!(y[Q..].iter().all(|&v| v == bits(15. / 4.)));
        let committed = s.accepted(2, 1).unwrap();
        assert_eq!(committed.len, 3);
        assert_eq!(committed.v.codes.len(), 3 * KV);
        assert_eq!(committed.v.value(2 * KV), bits(4.));
        assert!(s.accepted(3, 2).is_err());
        let z = gate(&y, &vec![0; y.len()]).unwrap();
        for (i, &v) in z.iter().enumerate() {
            assert_eq!(v, bits(num(y[i]) * 0.5));
        }
    }
}
