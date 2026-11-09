use anyhow::{Context, Result, bail};
use qwen_hybrid::{embedding::RawEmbeddings, exl3, projection::Pack, weights};
use std::io::Write;
use std::path::Path;

fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("embedding-rows") if a.len() == 5 => {
            anyhow::ensure!(a[3].len() <= 512, "bounded token list");
            let tokens = a[3]
                .split(',')
                .map(str::parse::<usize>)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let result = RawEmbeddings::load(&Pack::open(Path::new(&a[2]))?, &tokens)?;
            let raw = result
                .values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&a[4])?
                .write_all(&raw)?;
            println!("{}", serde_json::to_string_pretty(&result.provenance)?);
        }
        Some("inspect") if a.len() == 3 => println!(
            "{}",
            serde_json::to_string_pretty(&weights::inspect(Path::new(&a[2]))?)?
        ),
        Some("decode-tiles") if a.len() == 5 => {
            let window = a[2].parse::<u8>()?;
            anyhow::ensure!((8..=16).contains(&window), "invalid window");
            let data = std::fs::read(&a[3])?;
            anyhow::ensure!(
                data.len() <= 1 << 20 && data.len() % exl3::TILE_BYTES == 0,
                "invalid bounded tile fixture"
            );
            let mut out = Vec::new();
            for b in data.chunks_exact(exl3::TILE_BYTES) {
                for v in exl3::decode_k4_tile(b.try_into().unwrap(), window) {
                    out.extend(v.to_le_bytes());
                }
            }
            std::fs::write(&a[4], out)?;
        }
        Some("decode-mcg-tiles") if a.len() == 6 => {
            let n = a[2].parse::<usize>()?;
            let window = a[3].parse::<u8>()?;
            anyhow::ensure!((24..=128).contains(&n) && n % 2 == 0, "invalid rate");
            anyhow::ensure!((12..=16).contains(&window), "invalid window");
            let size = std::fs::metadata(&a[4])?.len();
            anyhow::ensure!(
                size <= 1 << 20 && size % (n * 2) as u64 == 0,
                "invalid bounded tile fixture"
            );
            let data = std::fs::read(&a[4])?;
            anyhow::ensure!(data.len() as u64 == size, "fixture changed");
            let mut out = Vec::new();
            for tile in data.chunks_exact(n * 2) {
                for v in exl3::decode_mcg_tile(tile, n, window)? {
                    out.extend(v.to_le_bytes());
                }
            }
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&a[5])?
                .write_all(&out)?;
        }
        Some("codebook") if a.len() == 4 => {
            let window = a[2].parse::<u8>()?;
            anyhow::ensure!((8..=16).contains(&window), "invalid window");
            let out: Vec<_> = (0..=u16::MAX)
                .flat_map(|c| exl3::mcg_bits(c, window).to_le_bytes())
                .collect();
            std::fs::write(a.get(3).context("output path")?, out)?;
        }
        _ => bail!(
            "usage: qwen-hybrid inspect MODEL | embedding-rows MODEL TOKEN_CSV OUTPUT | codebook WINDOW OUTPUT | decode-tiles WINDOW INPUT OUTPUT | decode-mcg-tiles HALFWORDS WINDOW INPUT OUTPUT"
        ),
    }
    Ok(())
}
