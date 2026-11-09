//! Bounded real-input fixtures for R05c. No GPU access in this module.
//! Prepared operands come from the qualified R05b layer-0 reference extraction.
use crate::{
    projection::sha,
    tree_scan::{self, Operands, Topology},
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{fs, path::Path};

pub const HEADS: usize = 48;
pub const DIM: usize = 128;
pub const STATE: usize = HEADS * DIM * DIM;
pub const MAX_ROWS: usize = 16;
pub const MANIFEST_SHA256: &str =
    "02cbd83196a73750ebd5ccceb4dbaf77e3d3ede0e9be1d0bf46bc98c87cc9ee1";
#[derive(Deserialize)]
struct Manifest {
    experiment: String,
    passed: bool,
    gpu_opened: bool,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    prompt: String,
    chunk: usize,
    heads: usize,
    shape_per_head: [usize; 3],
    payloads: Vec<Payload>,
}
#[derive(Deserialize)]
struct Payload {
    file: String,
    bytes: usize,
    sha256: String,
}

#[derive(Clone)]
pub struct Fixture {
    pub name: String,
    pub rows: usize,
    pub initial: Vec<f32>,
    /// All vectors are [head, token, channel]. Scalars are [head, token].
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub decay: Vec<f32>,
    pub beta: Vec<f32>,
}
impl Fixture {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&self.rows),
            "R05c rows must be 1..16"
        );
        ensure!(self.initial.len() == STATE, "initial state shape");
        for x in [&self.q, &self.k, &self.v] {
            ensure!(x.len() == HEADS * self.rows * DIM, "prepared vector shape");
        }
        for x in [&self.decay, &self.beta] {
            ensure!(x.len() == HEADS * self.rows, "prepared scalar shape");
        }
        ensure!(
            self.inputs()
                .iter()
                .all(|x| x.iter().all(|v| v.is_finite())),
            "nonfinite prepared input"
        );
        ensure!(
            self.decay
                .iter()
                .chain(&self.beta)
                .all(|v| (0.0..=1.0).contains(v)),
            "gate outside [0,1]"
        );
        Ok(())
    }
    pub fn inputs(&self) -> [&[f32]; 6] {
        [
            &self.initial,
            &self.q,
            &self.k,
            &self.v,
            &self.decay,
            &self.beta,
        ]
    }
    /// Take a sequence window with explicit state ownership. Head strides are
    /// repacked; taking a flat prefix of a head-major array would be incorrect.
    pub fn window(&self, start: usize, rows: usize, initial: &[f32]) -> Result<Self> {
        self.validate()?;
        ensure!(
            rows > 0 && rows <= self.rows && start <= self.rows - rows,
            "window bounds"
        );
        let slice = |x: &[f32], channels: usize| -> Vec<f32> {
            (0..HEADS)
                .flat_map(|h| {
                    x[(h * self.rows + start) * channels..(h * self.rows + start + rows) * channels]
                        .iter()
                        .copied()
                })
                .collect()
        };
        let result = Self {
            name: format!("{}:{start}+{rows}", self.name),
            rows,
            initial: initial.to_vec(),
            q: slice(&self.q, DIM),
            k: slice(&self.k, DIM),
            v: slice(&self.v, DIM),
            decay: slice(&self.decay, 1),
            beta: slice(&self.beta, 1),
        };
        result.validate()?;
        Ok(result)
    }
    pub fn prefix(&self, rows: usize) -> Result<Self> {
        self.window(0, rows, &self.initial)
    }
    pub fn receipt(&self) -> serde_json::Value {
        let hashes: Vec<_> = self
            .inputs()
            .iter()
            .map(|x| sha(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()))
            .collect();
        serde_json::json!({"name":self.name,"rows":self.rows,"heads":HEADS,"dimension":DIM,"input_sha256_in_binding_order":hashes})
    }
}

pub fn load_fixtures(root: &Path) -> Result<Vec<Fixture>> {
    let raw = fs::read(root.join("result.json"))?;
    ensure!(sha(&raw) == MANIFEST_SHA256, "R05b manifest pin mismatch");
    let manifest: Manifest = serde_json::from_slice(&raw)?;
    ensure!(
        manifest.experiment == "R05b"
            && manifest.passed
            && !manifest.gpu_opened
            && manifest.cases.len() == 4,
        "qualified R05b receipt"
    );
    let mut fixtures = Vec::new();
    for c in manifest.cases {
        ensure!(
            ["code-01", "prose-01"].contains(&c.prompt.as_str())
                && c.chunk < 2
                && c.heads == HEADS
                && c.shape_per_head == [MAX_ROWS, DIM, DIM],
            "R05b geometry/name"
        );
        let name = format!("{}-chunk{}", c.prompt, c.chunk);
        let mut values = Vec::new();
        ensure!(c.payloads.len() == 6, "R05b payload count");
        for (filename, n) in [
            ("initial.f32", STATE),
            ("q.f32", HEADS * MAX_ROWS * DIM),
            ("k.f32", HEADS * MAX_ROWS * DIM),
            ("v.f32", HEADS * MAX_ROWS * DIM),
            ("decay.f32", HEADS * MAX_ROWS),
            ("beta.f32", HEADS * MAX_ROWS),
        ] {
            let entries: Vec<_> = c.payloads.iter().filter(|p| p.file == filename).collect();
            ensure!(entries.len() == 1, "unique payload entry");
            let p = entries[0];
            let path = root.join(&name).join(filename);
            ensure!(
                p.bytes == n * 4 && fs::metadata(&path)?.len() == (n * 4) as u64,
                "payload size before read"
            );
            let data = fs::read(path)?;
            ensure!(sha(&data) == p.sha256, "payload hash {name}/{filename}");
            values.push(
                data.chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            );
        }
        let [initial, q, k, v, decay, beta]: [Vec<f32>; 6] =
            values.try_into().ok().context("six operands")?;
        let fixture = Fixture {
            name,
            rows: MAX_ROWS,
            initial,
            q,
            k,
            v,
            decay,
            beta,
        };
        fixture.validate()?;
        fixtures.push(fixture);
    }
    Ok(fixtures)
}

pub struct Reference {
    pub y: Vec<f64>,
    /// [token, head, value, key], independent FP64 scalar recurrence.
    pub snapshots: Vec<f64>,
}
pub fn reference(f: &Fixture) -> Result<Reference> {
    f.validate()?;
    let mut y = vec![0.; HEADS * f.rows * DIM];
    let mut snapshots = vec![0.; f.rows * STATE];
    for h in 0..HEADS {
        let vector = |x: &[f32], n: usize| {
            x[h * n..(h + 1) * n]
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        };
        let x = Operands {
            topology: Topology::new((0..f.rows).map(|t| t.checked_sub(1)).collect())?,
            dk: DIM,
            dv: DIM,
            initial: vector(&f.initial, DIM * DIM),
            q: vector(&f.q, f.rows * DIM),
            k: vector(&f.k, f.rows * DIM),
            v: vector(&f.v, f.rows * DIM),
            decay: vector(&f.decay, f.rows),
            beta: vector(&f.beta, f.rows),
        };
        let r = tree_scan::explicit::<f64>(&x)?;
        y[h * f.rows * DIM..(h + 1) * f.rows * DIM].copy_from_slice(&r.outputs);
        for (t, state) in r.states.iter().enumerate() {
            snapshots[t * STATE + h * DIM * DIM..t * STATE + (h + 1) * DIM * DIM]
                .copy_from_slice(state);
        }
    }
    ensure!(
        y.iter().chain(&snapshots).all(|x| x.is_finite()),
        "finite FP64 reference"
    );
    Ok(Reference { y, snapshots })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn patterned() -> Fixture {
        let n = HEADS * MAX_ROWS * DIM;
        Fixture {
            name: "stride-control".into(),
            rows: MAX_ROWS,
            initial: vec![0.; STATE],
            q: (0..n).map(|i| i as f32).collect(),
            k: vec![0.; n],
            v: vec![0.; n],
            decay: vec![1.; HEADS * MAX_ROWS],
            beta: vec![0.; HEADS * MAX_ROWS],
        }
    }
    #[test]
    fn window_repackages_every_head_and_owns_committed_state() {
        let f = patterned();
        let initial = vec![0.25; STATE];
        let w = f.window(3, 8, &initial).unwrap();
        for h in 0..HEADS {
            for t in 0..8 {
                for c in 0..DIM {
                    assert_eq!(w.q[(h * 8 + t) * DIM + c], f.q[(h * 16 + t + 3) * DIM + c]);
                }
            }
        }
        assert_eq!(w.initial, initial);
        assert!(f.initial.iter().all(|&v| v == 0.));
        assert!(f.window(9, 8, &initial).is_err());
        assert!(f.prefix(0).is_err());
        assert!(f.window(0, 8, &[]).is_err());
    }
    #[test]
    fn malformed_operands_fail_before_dispatch() {
        let mut f = patterned();
        f.q[9] = f32::NAN;
        assert!(f.validate().is_err());
        f.q[9] = 0.;
        f.decay[2] = 1.01;
        assert!(f.validate().is_err());
        f.decay[2] = 1.;
        f.beta.pop();
        assert!(f.validate().is_err());
    }
}
