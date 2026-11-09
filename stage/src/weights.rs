//! Read only Safetensors metadata or explicitly bounded tensor ranges.
//! Model inspection never maps/uploads all weights or starts a runtime.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
};

const MAX_HEADER: u64 = 16 << 20;
fn direct_reads() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_DIRECT_READS").as_deref() != Ok("0"))
}
const MAX_RANGE: u64 = 64 << 20;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tensor {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}
impl Tensor {
    pub fn bytes(&self) -> u64 {
        self.data_offsets[1] - self.data_offsets[0]
    }
    fn check(&self) -> Result<()> {
        let size = match self.dtype.as_str() {
            "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1_u64,
            "U16" | "I16" | "F16" | "BF16" => 2,
            "U32" | "I32" | "F32" => 4,
            "U64" | "I64" | "F64" => 8,
            d => bail!("unsupported dtype {d}"),
        };
        let bytes = self
            .shape
            .iter()
            .try_fold(size, |n, &s| n.checked_mul(s))
            .context("tensor byte count overflow")?;
        ensure!(
            self.data_offsets[0] <= self.data_offsets[1],
            "reversed tensor range"
        );
        ensure!(bytes == self.bytes(), "shape/dtype byte count mismatch");
        Ok(())
    }
}

#[derive(Debug)]
pub struct Shard {
    pub path: PathBuf,
    pub data_start: u64,
    pub file_bytes: u64,
    pub header_sha256: String,
    pub metadata: BTreeMap<String, String>,
    pub tensors: BTreeMap<String, Tensor>,
}
impl Shard {
    pub fn open(path: &Path) -> Result<Self> {
        let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let file_bytes = f.metadata()?.len();
        let mut prefix = [0_u8; 8];
        f.read_exact(&mut prefix)?;
        let n = u64::from_le_bytes(prefix);
        ensure!(
            (2..=MAX_HEADER).contains(&n),
            "invalid or excessive header length"
        );
        let data_start = n.checked_add(8).context("header length overflow")?;
        ensure!(data_start <= file_bytes, "header exceeds file");
        let mut bytes = vec![0; n as usize];
        f.read_exact(&mut bytes)?;
        let mut raw: BTreeMap<String, Value> = serde_json::from_slice(&bytes)?;
        let metadata = raw
            .remove("__metadata__")
            .map(serde_json::from_value)
            .transpose()?
            .unwrap_or_default();
        let mut tensors = BTreeMap::new();
        for (name, v) in raw {
            let tensor: Tensor = serde_json::from_value(v).with_context(|| name.clone())?;
            tensor.check().with_context(|| name.clone())?;
            ensure!(
                tensor.data_offsets[1] <= file_bytes - data_start,
                "tensor {name} exceeds file"
            );
            tensors.insert(name, tensor);
        }
        let mut ranges: Vec<_> = tensors.values().map(|t| t.data_offsets).collect();
        ranges.sort_unstable();
        let mut end = 0;
        for [lo, hi] in ranges {
            ensure!(lo == end, "overlapping or non-contiguous tensor ranges");
            end = hi;
        }
        ensure!(end == file_bytes - data_start, "unclaimed shard data");
        Ok(Self {
            path: path.to_owned(),
            data_start,
            file_bytes,
            header_sha256: format!("{:x}", Sha256::digest(&bytes)),
            metadata,
            tensors,
        })
    }

    pub fn read_range(&self, name: &str, offset: u64, length: u64) -> Result<Vec<u8>> {
        let t = self.tensors.get(name).context("unknown tensor")?;
        ensure!(length <= MAX_RANGE, "range exceeds bounded-reader limit");
        ensure!(
            offset.checked_add(length).is_some_and(|n| n <= t.bytes()),
            "tensor read out of bounds"
        );
        let mut f = File::open(&self.path)?;
        ensure!(
            f.metadata()?.len() == self.file_bytes,
            "shard changed since inspection"
        );
        let at = self.data_start + t.data_offsets[0] + offset;
        // R56: bulk weight reads (>= 1 MiB) bypass the page cache (page-aligned direct reads):
        // the weights are copied to device memory and their cached pages would only crowd the
        // n-gram tables and the Metal target's mapped experts. QWEN_DIRECT_READS=0 disables.
        if length >= 1 << 20 && direct_reads() {
            return Ok(crate::io_pool::read_uncached(&self.path, at, length as usize)?);
        }
        f.seek(SeekFrom::Start(at))?;
        let mut data = vec![0; length as usize];
        f.read_exact(&mut data)?;
        Ok(data)
    }
}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

#[derive(Serialize)]
pub struct Inventory {
    pub config_sha256: String,
    pub index_sha256: String,
    pub shards: usize,
    pub tensors: usize,
    pub tensor_bytes: u64,
    pub trunk_expert_bytes: u64,
    pub shard_headers: BTreeMap<String, String>,
    pub backend_opened: bool,
}

/// Deliberately admit only the initial model/pack contract. Expand this when an
/// additional configuration has its own evidence, not via a silent fallback.
pub fn check_config(c: &Value) -> Result<()> {
    ensure!(c["model_type"] == "qwen4_exp", "expected Qwen Flash Next");
    let t = &c["text_config"];
    for (k, v) in [
        ("hidden_size", 2560),
        ("moe_intermediate_size", 640),
        ("num_hidden_layers", 48),
        ("num_experts", 512),
        ("num_experts_per_tok", 10),
        ("hc_count", 4),
        ("max_position_embeddings", 262144),
        ("vocab_size", 248320),
    ] {
        ensure!(t[k] == v, "unsupported {k}");
    }
    ensure!(t["mamba_ssm_dtype"] == "float32", "GDN state must be FP32");
    ensure!(
        t["rope_parameters"]["rope_type"] == "default",
        "use the native-RoPE view"
    );
    let layers = t["layer_types"].as_array().context("missing layer types")?;
    ensure!(layers.len() == 48, "layer count mismatch");
    for (i, l) in layers.iter().enumerate() {
        ensure!(
            *l == if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            },
            "layer type mismatch at {i}"
        );
    }
    let e = &c["expert_quant"];
    ensure!(
        e["format"] == "exl3"
            && e["k"] == 4
            && e["codebook"] == "mcg"
            && e["window"] == 15
            && e["out_scales"] == "svh",
        "unsupported EXL3 pack"
    );
    ensure!(
        c["ngram_table"]["bits"] == 16,
        "expected original BF16 n-gram table"
    );
    Ok(())
}

fn filename_only(s: &str) -> bool {
    let mut it = Path::new(s).components();
    matches!(it.next(), Some(Component::Normal(_))) && it.next().is_none()
}

pub fn inspect(root: &Path) -> Result<Inventory> {
    let config_bytes = std::fs::read(root.join("config.json"))?;
    let config: Value = serde_json::from_slice(&config_bytes)?;
    check_config(&config)?;
    let index_bytes = std::fs::read(root.join("model.safetensors.index.json"))?;
    let index: Index = serde_json::from_slice(&index_bytes)?;
    let files: BTreeSet<_> = index.weight_map.values().collect();
    let mut found = BTreeMap::new();
    let mut header_hashes = BTreeMap::new();
    let mut all_bytes = 0_u64;
    let mut expert_bytes = 0_u64;
    let mut expert_parts = 0_usize;
    for file in &files {
        ensure!(filename_only(file), "index path is not a file name");
        let shard = Shard::open(&root.join(file))?;
        header_hashes.insert((*file).clone(), shard.header_sha256.clone());
        for (name, tensor) in &shard.tensors {
            ensure!(
                index.weight_map.get(name) == Some(file),
                "index/shard mismatch for {name}"
            );
            ensure!(
                found.insert(name.clone(), (*file).clone()).is_none(),
                "duplicate tensor {name}"
            );
            all_bytes = all_bytes
                .checked_add(tensor.bytes())
                .context("inventory bytes overflow")?;
            if name.starts_with("language_model.model.layers.") && name.contains(".mlp.switch_mlp.")
            {
                check_expert(name, tensor, &shard.metadata)?;
                expert_bytes += tensor.bytes();
                expert_parts += 1;
            }
        }
    }
    ensure!(found == index.weight_map, "missing indexed tensor");
    ensure!(
        expert_parts == 48 * 9,
        "expected three complete EXL3 projections per trunk layer"
    );
    // Validate each required name, not just a count that could conceal a missing layer.
    for l in 0..48 {
        for p in ["gate_proj", "up_proj", "down_proj"] {
            for part in ["trellis", "suh", "svh"] {
                ensure!(
                    found.contains_key(&format!(
                        "language_model.model.layers.{l}.mlp.switch_mlp.{p}.{part}"
                    )),
                    "missing expert part"
                );
            }
        }
    }
    Ok(Inventory {
        config_sha256: format!("{:x}", Sha256::digest(config_bytes)),
        index_sha256: format!("{:x}", Sha256::digest(index_bytes)),
        shards: files.len(),
        tensors: found.len(),
        tensor_bytes: all_bytes,
        trunk_expert_bytes: expert_bytes,
        shard_headers: header_hashes,
        backend_opened: false,
    })
}

fn check_expert(name: &str, t: &Tensor, metadata: &BTreeMap<String, String>) -> Result<()> {
    for (k, v) in [
        ("format", "exl3"),
        ("k", "4"),
        ("codebook", "mcg"),
        ("window", "15"),
    ] {
        ensure!(
            metadata.get(k).map(String::as_str) == Some(v),
            "EXL3 shard stamp mismatch: {k}"
        );
    }
    let (input, output) = if name.contains(".down_proj.") {
        (640, 2560)
    } else {
        (2560, 640)
    };
    let (shape, dtype) = if name.ends_with(".trellis") {
        (vec![512, input / 16, output / 16, 64], "U16")
    } else if name.ends_with(".suh") {
        (vec![512, input], "F16")
    } else if name.ends_with(".svh") {
        (vec![512, output], "F16")
    } else {
        bail!("unknown EXL3 leaf {name}")
    };
    ensure!(
        t.shape == shape && t.dtype == dtype,
        "EXL3 geometry mismatch for {name}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_bad_sizes_and_paths() {
        for s in ["../a", "/a", "x/y", "", "."] {
            assert!(!filename_only(s));
        }
        assert!(filename_only("model-001.safetensors"));
        let t = Tensor {
            dtype: "F16".into(),
            shape: vec![u64::MAX, 2],
            data_offsets: [0, 2],
        };
        assert!(t.check().is_err());
        let t = Tensor {
            dtype: "F16".into(),
            shape: vec![1],
            data_offsets: [5, 3],
        };
        assert!(t.check().is_err());
    }
}
