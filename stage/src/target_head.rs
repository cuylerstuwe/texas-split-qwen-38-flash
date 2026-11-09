//! CPU vocabulary-head reference, scanned in bounded affine8 tiles.
//! The original target head is shared with DFlash; no vocabulary pruning.
use crate::{
    dense::Affine8,
    embedding::{HIDDEN, VOCAB},
    projection::Pack,
    rounding::bf16_from_f64,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug)]
pub enum WeightArithmetic {
    OriginalAffine8,
    /// Explicit proposal-only approximation for a dense CUDA prototype.
    DecodedBf16,
}

pub struct Reference {
    pub logits: Vec<u16>,
    pub provenance: Value,
}
pub fn reference(pack: &Pack, hidden: &[u16]) -> Result<Reference> {
    reference_with_arithmetic(pack, hidden, WeightArithmetic::OriginalAffine8)
}
pub fn reference_with_arithmetic(
    pack: &Pack,
    hidden: &[u16],
    arithmetic: WeightArithmetic,
) -> Result<Reference> {
    ensure!(
        !hidden.is_empty() && hidden.len() % HIDDEN == 0 && hidden.len() / HIDDEN <= 16,
        "bounded head input"
    );
    let rows = hidden.len() / HIDDEN;
    let mut logits = vec![0; rows * VOCAB];
    let mut tiles = vec![];
    for first in (0..VOCAB).step_by(2048) {
        let n = (VOCAB - first).min(2048);
        let (w, codes, wp) = pack.tensor_row_range("language_model.lm_head.weight", first, n)?;
        let (s, sb, sp) = pack.tensor_row_range("language_model.lm_head.scales", first, n)?;
        let (b, bb, bp) = pack.tensor_row_range("language_model.lm_head.biases", first, n)?;
        ensure!(
            w.dtype == "U32" && w.shape == [VOCAB as u64, (HIDDEN / 4) as u64],
            "target head affine8 geometry"
        );
        ensure!(
            s.dtype == "BF16"
                && b.dtype == "BF16"
                && s.shape == b.shape
                && s.shape == [VOCAB as u64, 20],
            "target head group128 metadata"
        );
        let words = |raw: &[u8]| {
            raw.chunks_exact(2)
                .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
                .collect()
        };
        let tile = Affine8 {
            k: HIDDEN,
            n,
            group: 128,
            codes,
            scales: words(&sb),
            biases: words(&bb),
            provenance: json!([wp, sp, bp]),
        };
        tile.validate()?;
        let mut decoded = tile.oracle_weights()?;
        if matches!(arithmetic, WeightArithmetic::DecodedBf16) {
            for w in &mut decoded {
                *w = bf16_from_f64(f64::from(*w)).to_f32();
            }
        }
        let values = tile.reference(&decoded, hidden)?;
        for r in 0..rows {
            for j in 0..n {
                logits[r * VOCAB + first + j] = bf16_from_f64(values[r * n + j]).to_bits();
            }
        }
        tiles.push(tile.provenance);
    }
    Ok(Reference {
        logits,
        provenance: json!({"kind":match arithmetic { WeightArithmetic::OriginalAffine8=>"original_target_affine8_head", WeightArithmetic::DecodedBf16=>"proposal_head_decoded_bf16_approximation" },"vocab":VOCAB,"hidden":HIDDEN,"rows":rows,"tile_rows":2048,"group":128,"tiles":tiles,"arithmetic":match arithmetic { WeightArithmetic::OriginalAffine8=>"FP64 dot over FP32 decoded affine8, BF16 output; no multiplier or softcap declared by this checkpoint",WeightArithmetic::DecodedBf16=>"affine8 decoded FP32 then BF16 weights, FP64 dot, BF16 output; proposal approximation, target unchanged" },"gpu_opened":false,"not_a_gpu_speed_measurement":true}),
    })
}

/// R57: the whole affine8 head decoded once to FP32 [VOCAB][HIDDEN] (the same per-weight decode
/// as `reference`), for screens that evaluate many rows (2.5 GB).
pub fn decoded_fp32(pack: &Pack) -> Result<Vec<f32>> {
    let mut w = vec![0f32; VOCAB * HIDDEN];
    for first in (0..VOCAB).step_by(2048) {
        let n = (VOCAB - first).min(2048);
        let (wt, codes, _) = pack.tensor_row_range("language_model.lm_head.weight", first, n)?;
        let (s, sb, _) = pack.tensor_row_range("language_model.lm_head.scales", first, n)?;
        let (b, bb, _) = pack.tensor_row_range("language_model.lm_head.biases", first, n)?;
        ensure!(
            wt.dtype == "U32" && s.dtype == "BF16" && b.dtype == "BF16" && s.shape == [VOCAB as u64, 20],
            "target head affine8 geometry"
        );
        let words = |raw: &[u8]| raw.chunks_exact(2).map(|v| u16::from_le_bytes(v.try_into().unwrap())).collect();
        let tile = Affine8 { k: HIDDEN, n, group: 128, codes, scales: words(&sb), biases: words(&bb), provenance: json!(null) };
        let d = tile.oracle_weights()?;
        ensure!(d.len() == n * HIDDEN, "head tile size");
        w[first * HIDDEN..(first + n) * HIDDEN].copy_from_slice(&d);
    }
    Ok(w)
}
