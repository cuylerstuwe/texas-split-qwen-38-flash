//! Sushi MCG tile decoding, including the original K4 and new fractional packs.
//!
//! Format and tile permutation follow Sushi / ExLlamaV3 (MIT); see NOTICE.txt.
//! The decoder is scalar truth for subsequent GPU implementations, not a fast
//! inference kernel. Weight values round to FP16 at the codebook boundary.
use anyhow::{Result, ensure};
use half::f16;

pub const TILE_BYTES: usize = 128;
pub const TILE_VALUES: usize = 256;

pub fn mcg_bits(codeword: u16, window: u8) -> u16 {
    assert!((8..=16).contains(&window));
    let mask = ((1_u32 << window) - 1) as u16;
    let hash = u32::from(codeword & mask).wrapping_mul(0xcbac_1fed);
    let pair = (hash & 0x8fff_8fff) ^ 0x3b60_3b60;
    let lo = f16::from_bits(pair as u16).to_f32();
    let hi = f16::from_bits((pair >> 16) as u16).to_f32();
    f16::from_f32(lo + hi).to_bits()
}

/// A K4 tile is 32 little-endian words, each supplying eight successive codes.
/// The first code in a word takes its high bits from the previous word; the
/// first word wraps to the tile's last word (tail biting).
pub fn decode_k4_tile(bytes: &[u8; TILE_BYTES], window: u8) -> [u16; TILE_VALUES] {
    let words: [u32; 32] =
        std::array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()));
    let mut result = [0_u16; TILE_VALUES];
    for lane in 0..32 {
        let stream = (u64::from(words[(lane + 31) % 32]) << 32) | u64::from(words[lane]);
        let r = (lane % 4) * 2;
        let c = lane / 4;
        for j in 0..8 {
            let row = r + (j % 2) + ((j % 4) / 2) * 8;
            let col = c + (j / 4) * 8;
            let cw = (stream >> (28 - j * 4)) as u16;
            result[row * 16 + col] = mcg_bits(cw, window);
        }
    }
    result
}

/// Fractional-rate MCG tile. `halfwords=28` is K1.75: each weight advances
/// floor((t+1)*28/16)-floor(t*28/16) bits, not a truncated K4 bitstream.
/// Uses paired funnel loads; the slow bit-reader tests below are independent.
pub fn decode_mcg_tile(bytes: &[u8], halfwords: usize, window: u8) -> Result<[u16; TILE_VALUES]> {
    ensure!(
        (24..=128).contains(&halfwords)
            && halfwords % 2 == 0
            && bytes.len() == halfwords * 2
            && (12..=16).contains(&window),
        "MCG tile rate/window/size"
    );
    let count = halfwords / 2;
    let total = halfwords * 16;
    let word = |i: usize| {
        u32::from_le_bytes(
            bytes[(i % count) * 4..(i % count + 1) * 4]
                .try_into()
                .unwrap(),
        )
    };
    let mut out = [0; TILE_VALUES];
    for pair in 0..128 {
        let e0 = (pair * 2 + 1) * halfwords / 16;
        let e1 = (pair * 2 + 2) * halfwords / 16;
        let a = (e0 + total - 16) / 32;
        let b = (e1 + total - 1) / 32;
        let shift = (b + 1) * 32 - (e1 + total);
        let merged = (u64::from(word(a)) << 32) | u64::from(word(b));
        let cw = (merged >> shift) as u32;
        for (j, c) in [cw >> (e1 - e0), cw].into_iter().enumerate() {
            let t = pair * 2 + j;
            let lane = t / 8;
            let k = t % 8;
            let row = (lane % 4) * 2 + [0, 1, 8, 9][k % 4];
            let col = lane / 4 + if k >= 4 { 8 } else { 0 };
            out[row * 16 + col] = mcg_bits(c as u16, window);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn k175_two_word_lane_window_matches_independent_full_tile() {
        let mut seed = 0x312ab70du32;
        for trial in 0..258 {
            let bytes: Vec<u8> = (0..56)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    if trial == 0 {
                        0
                    } else if trial == 1 {
                        255
                    } else {
                        seed as u8
                    }
                })
                .collect();
            let reference = decode_mcg_tile(&bytes, 28, 15).unwrap();
            let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
            for lane in 0..32 {
                let end = lane * 14 + 14;
                let last = (end - 1) / 32;
                let prev = if last == 0 { 13 } else { last - 1 };
                let window =
                    (((word(prev) as u64) << 32) | word(last) as u64) >> ((last + 1) * 32 - end);
                for (pair, offset) in [11, 7, 4, 0].into_iter().enumerate() {
                    let cw = (window >> offset) as u32;
                    for (j, c) in [cw >> 2, cw].into_iter().enumerate() {
                        let t = pair * 2 + j;
                        let row = (lane % 4) * 2 + [0, 1, 8, 9][t % 4];
                        let col = lane / 4 + if t >= 4 { 8 } else { 0 };
                        assert_eq!(
                            mcg_bits(c as u16, 15),
                            reference[row * 16 + col],
                            "trial{trial} lane{lane} pair{pair} j{j}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn fractional_mcg_matches_independent_bit_reader_and_existing_k4() {
        let mut state = 0x12345678u32;
        for n in (24..=128).step_by(2) {
            for window in [12, 15, 16] {
                for _ in 0..8 {
                    let bytes: Vec<_> = (0..2 * n)
                        .map(|_| {
                            state ^= state << 13;
                            state ^= state >> 17;
                            state ^= state << 5;
                            state as u8
                        })
                        .collect();
                    let got = decode_mcg_tile(&bytes, n, window).unwrap();
                    for t in 0..256 {
                        let end = (t + 1) * n / 16;
                        let mut code = 0u16;
                        for bit in 0..16 {
                            let pos = (end + n * 16 - 1 - bit) % (n * 16);
                            let b = (pos / 32) * 4 + (31 - pos % 32) / 8;
                            code |= u16::from((bytes[b] >> ((31 - pos % 32) % 8)) & 1) << bit;
                        }
                        let lane = t / 8;
                        let j = t % 8;
                        let row = (lane % 4) * 2 + [0, 1, 8, 9][j % 4];
                        let col = lane / 4 + if j >= 4 { 8 } else { 0 };
                        assert_eq!(
                            got[row * 16 + col],
                            mcg_bits(code, window),
                            "n{n} window{window} t{t}"
                        );
                    }
                    if n == 64 {
                        assert_eq!(
                            got,
                            decode_k4_tile(bytes.as_slice().try_into().unwrap(), window)
                        );
                    }
                }
            }
        }
        assert!(decode_mcg_tile(&[0; 56], 27, 15).is_err());
        assert!(decode_mcg_tile(&[0; 56], 28, 11).is_err());
        assert!(decode_mcg_tile(&[0; 55], 28, 15).is_err());
        assert_eq!(
            (0..8)
                .map(|t| (t + 1) * 28 / 16 - t * 28 / 16)
                .collect::<Vec<_>>(),
            [1, 2, 2, 2, 1, 2, 2, 2]
        );
    }
    #[test]
    fn window_is_a_weight_parameter_not_a_performance_flag() {
        for c in 0..=u16::MAX {
            assert_eq!(mcg_bits(c, 15), mcg_bits(c & 0x7fff, 16));
            assert!(f16::from_bits(mcg_bits(c, 15)).is_finite());
        }
        assert!((0x8000..=u16::MAX).any(|c| mcg_bits(c, 15) != mcg_bits(c, 16)));
    }

    #[test]
    fn tile_matches_bit_by_bit_tail_biting_reference() {
        // Independent slow bit-indexed reader: no funnel/adjacent-word logic.
        let mut state = 0x9e37_79b9_u32;
        for _ in 0..64 {
            let bytes = std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            });
            let result = decode_k4_tile(&bytes, 15);
            let mut touched = [false; 256];
            for t in 0..256 {
                let end = (t + 1) * 4;
                let mut cw = 0_u16;
                for bit in 0..16 {
                    let pos = (end + 1024 - 1 - bit) % 1024;
                    // Words advance in memory, bits within each word go MSB first.
                    let word = pos / 32;
                    let in_word = 31 - pos % 32;
                    cw |= u16::from((bytes[word * 4 + in_word / 8] >> (in_word % 8)) & 1) << bit;
                }
                let lane = t / 8;
                let j = t % 8;
                let row = (lane % 4) * 2 + [0, 1, 8, 9][j % 4];
                let col = lane / 4 + if j >= 4 { 8 } else { 0 };
                let i = row * 16 + col;
                assert!(!touched[i]);
                touched[i] = true;
                assert_eq!(result[i], mcg_bits(cw, 15));
            }
            assert!(touched.into_iter().all(|x| x));
        }
    }
}
