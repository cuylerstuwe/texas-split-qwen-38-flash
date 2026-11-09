//! Direct FP64 -> BF16 round-to-nearest, ties-to-even for CPU reference math.
//! half 2.7.1's from_f64 discards the low 32 mantissa bits before its rounding
//! decision. Those bits can distinguish a midpoint from a value just above it.
//! Keep the complete significand here; GPU arithmetic/converters are separate.
use half::bf16;

const fn rounded_shift(value: u64, shift: u32) -> u64 {
    let high = value >> shift;
    let remainder = value & ((1u64 << shift) - 1);
    let halfway = 1u64 << (shift - 1);
    high + if remainder > halfway || (remainder == halfway && high & 1 != 0) {
        1
    } else {
        0
    }
}
pub const fn bf16_from_f64(value: f64) -> bf16 {
    let raw = value.to_bits();
    let sign = ((raw >> 48) & 0x8000) as u16;
    let exponent = ((raw >> 52) & 0x7ff) as i32;
    let fraction = raw & 0x000f_ffff_ffff_ffff;
    if exponent == 0x7ff {
        return bf16::from_bits(
            sign | 0x7f80
                | if fraction == 0 {
                    0
                } else {
                    ((fraction >> 45) as u16) | 0x40
                },
        );
    }
    // All FP64 subnormals are far below the smallest BF16 subnormal.
    if exponent == 0 {
        return bf16::from_bits(sign);
    }
    let e = exponent - 1023;
    let significand = (1u64 << 52) | fraction;
    if e > 127 {
        return bf16::from_bits(sign | 0x7f80);
    }
    if e >= -126 {
        // A rounded significand of256 naturally carries into the next exponent,
        // including the max-finite -> infinity boundary.
        return bf16::from_bits(
            sign | (((e + 126) as u16) * 128 + rounded_shift(significand, 45) as u16),
        );
    }
    if e < -134 {
        return bf16::from_bits(sign);
    }
    // BF16 subnormal spacing is2^-133; rounding may produce min-normal0x0080.
    bf16::from_bits(sign | rounded_shift(significand, (-e - 81) as u32) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_router_midpoint_keeps_discarded_bits() {
        let x = -7.265627647660949;
        assert_eq!(bf16_from_f64(x).to_f64(), -7.28125);
        assert_eq!(bf16_from_f64(-x).to_f64(), 7.28125);
        // Values below/above one exact midpoint, closer than a32-bit truncation.
        let m = 1.00390625f64;
        assert_eq!(
            bf16_from_f64(f64::from_bits(m.to_bits() - 1)).to_bits(),
            0x3f80
        );
        assert_eq!(
            bf16_from_f64(f64::from_bits(m.to_bits() + 1)).to_bits(),
            0x3f81
        );
    }
    #[test]
    fn every_finite_bfloat_midpoint_and_neighbor_rounds_correctly() {
        for low in 0u16..0x7f80 {
            let a = bf16::from_bits(low).to_f64();
            let b = if low == 0x7f7f {
                2f64.powi(128)
            } else {
                bf16::from_bits(low + 1).to_f64()
            };
            let middle = (a + b) * 0.5;
            let tie = if low & 1 == 0 { low } else { low + 1 };
            for (x, want) in [
                (f64::from_bits(middle.to_bits() - 1), low),
                (middle, tie),
                (f64::from_bits(middle.to_bits() + 1), low + 1),
            ] {
                assert_eq!(
                    bf16_from_f64(x).to_bits(),
                    want,
                    "positive midpoint {low:04x}"
                );
                assert_eq!(
                    bf16_from_f64(-x).to_bits(),
                    want | 0x8000,
                    "negative midpoint {low:04x}"
                );
            }
        }
    }
    #[test]
    fn all_bfloat_values_roundtrip_and_specials_are_preserved() {
        for bits in 0u16..=u16::MAX {
            let x = bf16::from_bits(bits);
            let result = bf16_from_f64(x.to_f64());
            if x.is_nan() {
                assert!(result.is_nan());
            } else {
                assert_eq!(result.to_bits(), bits);
            }
        }
        assert_eq!(bf16_from_f64(f64::from_bits(1)).to_bits(), 0);
        assert_eq!(bf16_from_f64(-f64::from_bits(1)).to_bits(), 0x8000);
        assert_eq!(bf16_from_f64(f64::MAX).to_bits(), 0x7f80);
        assert_eq!(bf16_from_f64(-f64::MAX).to_bits(), 0xff80);
    }
}
