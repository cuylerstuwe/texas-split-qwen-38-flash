//! CPU research oracle for branch-local Gated DeltaNet verification.
//!
//! Independently implemented from the recurrence and the TreeWY equations:
//! https://arxiv.org/html/2608.20961v2 (section 3).
//! State is [value, key]. Queries/keys, decay and beta are already prepared.
//! This does NOT implement Qwen's projections, convolution, HC, QSA, sampler,
//! or an optimized GPU triangular solve. Different reduction orders are allowed.

use anyhow::{Result, ensure};
use std::ops::{Add, Mul, Sub};

pub trait Scalar:
    Copy + Default + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    fn from_f64(x: f64) -> Self;
    fn as_f64(self) -> f64;
}
impl Scalar for f32 {
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn as_f64(self) -> f64 {
        self as f64
    }
}
impl Scalar for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    fn as_f64(self) -> f64 {
        self
    }
}

/// Parents precede children; None names the common committed prefix.
/// Several roots and arbitrary topological (not necessarily DFS) order work.
#[derive(Clone, Debug)]
pub struct Topology {
    parents: Vec<Option<usize>>,
}
impl Topology {
    pub fn new(parents: Vec<Option<usize>>) -> Result<Self> {
        ensure!(
            !parents.is_empty() && parents.len() <= 256,
            "research tree must have 1..256 nodes"
        );
        for (i, parent) in parents.iter().enumerate() {
            ensure!(
                parent.is_none_or(|p| p < i),
                "node {i}: parent must precede child"
            );
        }
        Ok(Self { parents })
    }
    pub fn len(&self) -> usize {
        self.parents.len()
    }
    pub fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }
    pub fn path(&self, leaf: usize) -> Result<Vec<usize>> {
        ensure!(leaf < self.len(), "invalid selected node");
        let mut path = vec![];
        let mut node = Some(leaf);
        while let Some(i) = node {
            path.push(i);
            node = self.parents[i];
        }
        path.reverse();
        Ok(path)
    }
}

#[derive(Clone, Debug)]
pub struct Operands {
    pub topology: Topology,
    pub dk: usize,
    pub dv: usize,
    pub initial: Vec<f64>,
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub decay: Vec<f64>,
    pub beta: Vec<f64>,
}
impl Operands {
    pub fn validate(&self) -> Result<()> {
        let n = self.topology.len();
        ensure!(
            (1..=256).contains(&self.dk) && (1..=256).contains(&self.dv),
            "research dimensions out of bounds"
        );
        ensure!(
            self.initial.len() == self.dk * self.dv
                && self.q.len() == n * self.dk
                && self.k.len() == n * self.dk
                && self.v.len() == n * self.dv
                && self.decay.len() == n
                && self.beta.len() == n,
            "operand shape mismatch"
        );
        ensure!(
            [
                &self.initial,
                &self.q,
                &self.k,
                &self.v,
                &self.decay,
                &self.beta
            ]
            .iter()
            .all(|x| x.iter().all(|y| y.is_finite())),
            "nonfinite input"
        );
        ensure!(
            self.decay.iter().all(|&a| (0.0..=1.0).contains(&a)),
            "decay outside [0,1]"
        );
        Ok(())
    }
}

fn dot<T: Scalar>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b)
        .fold(T::default(), |sum, (&x, &y)| sum + x * y)
}
fn cast<T: Scalar>(a: &[f64]) -> Vec<T> {
    a.iter().map(|&x| T::from_f64(x)).collect()
}

/// Explicit native-order rank-one update; separate from the compact algorithm.
fn step<T: Scalar>(state: &mut [T], k: &[T], v: &[T], a: T, beta: T) {
    for (value, row) in v.iter().zip(state.chunks_exact_mut(k.len())) {
        for s in row.iter_mut() {
            *s = *s * a;
        }
        let correction = beta * (*value - dot(row, k));
        for (s, &key) in row.iter_mut().zip(k) {
            *s = *s + correction * key;
        }
    }
}

pub struct Explicit<T> {
    pub outputs: Vec<T>,
    pub states: Vec<Vec<T>>,
}
/// Store-all reference, each node starts from its own parent's state.
pub fn explicit<T: Scalar>(x: &Operands) -> Result<Explicit<T>> {
    x.validate()?;
    let initial = cast(&x.initial);
    let q = cast(&x.q);
    let k = cast(&x.k);
    let v = cast(&x.v);
    let mut states: Vec<Vec<T>> = vec![];
    let mut outputs = vec![];
    for i in 0..x.topology.len() {
        let mut s = x.topology.parents[i].map_or_else(|| initial.clone(), |p| states[p].clone());
        step(
            &mut s,
            &k[i * x.dk..(i + 1) * x.dk],
            &v[i * x.dv..(i + 1) * x.dv],
            T::from_f64(x.decay[i]),
            T::from_f64(x.beta[i]),
        );
        for row in s.chunks_exact(x.dk) {
            outputs.push(dot(row, &q[i * x.dk..(i + 1) * x.dk]));
        }
        states.push(s);
    }
    Ok(Explicit { outputs, states })
}

/// Replay only the selected path, without ever consulting another branch.
/// Later GPU code can retain rounded operands and use native update arithmetic.
pub fn replay<T: Scalar>(x: &Operands, leaf: Option<usize>) -> Result<Vec<T>> {
    x.validate()?;
    let mut state = cast(&x.initial);
    if let Some(leaf) = leaf {
        for i in x.topology.path(leaf)? {
            step(
                &mut state,
                &cast::<T>(&x.k[i * x.dk..(i + 1) * x.dk]),
                &cast::<T>(&x.v[i * x.dv..(i + 1) * x.dv]),
                T::from_f64(x.decay[i]),
                T::from_f64(x.beta[i]),
            );
        }
    }
    Ok(state)
}

/// Retains one pseudo-value vector per node, plus scalar ancestor-decay weights.
/// Retained operands and base state are included here for transparent accounting.
pub struct Compact<T> {
    pub outputs: Vec<T>,
    pub pseudo_values: Vec<T>,
    ancestor_decay: Vec<T>,
    base_decay: Vec<T>,
    initial: Vec<T>,
    keys: Vec<T>,
    topology: Topology,
    dk: usize,
    dv: usize,
}
impl<T: Scalar> Compact<T> {
    pub fn reconstruct(&self, leaf: Option<usize>) -> Result<Vec<T>> {
        let Some(i) = leaf else {
            return Ok(self.initial.clone());
        };
        let path = self.topology.path(i)?;
        let n = self.topology.len();
        let mut s: Vec<T> = self
            .initial
            .iter()
            .map(|&v| self.base_decay[i] * v)
            .collect();
        for j in path {
            for d in 0..self.dv {
                let u = self.ancestor_decay[i * n + j] * self.pseudo_values[j * self.dv + d];
                for c in 0..self.dk {
                    s[d * self.dk + c] = s[d * self.dk + c] + u * self.keys[j * self.dk + c];
                }
            }
        }
        Ok(s)
    }
    pub fn retained_scalar_count(&self) -> usize {
        self.outputs.len()
            + self.pseudo_values.len()
            + self.ancestor_decay.len()
            + self.base_decay.len()
            + self.initial.len()
            + self.keys.len()
    }
}

pub fn compact<T: Scalar>(x: &Operands) -> Result<Compact<T>> {
    x.validate()?;
    let n = x.topology.len();
    let q: Vec<T> = cast(&x.q);
    let k: Vec<T> = cast(&x.k);
    let v: Vec<T> = cast(&x.v);
    let s0: Vec<T> = cast(&x.initial);
    // Products along edges avoid division by tiny/zero cumulative decays.
    let mut ancestor = vec![T::default(); n * n];
    let mut base = vec![T::default(); n];
    for i in 0..n {
        let mut factor = T::from_f64(1.0);
        let mut current = Some(i);
        while let Some(j) = current {
            ancestor[i * n + j] = factor;
            factor = factor * T::from_f64(x.decay[j]);
            current = x.topology.parents[j];
        }
        base[i] = factor;
    }
    let mut u = vec![T::default(); n * x.dv];
    let mut outputs = vec![T::default(); n * x.dv];
    for i in 0..n {
        let ki = &k[i * x.dk..(i + 1) * x.dk];
        let qi = &q[i * x.dk..(i + 1) * x.dk];
        let b = T::from_f64(x.beta[i]);
        for d in 0..x.dv {
            u[i * x.dv + d] =
                b * (v[i * x.dv + d] - base[i] * dot(&s0[d * x.dk..(d + 1) * x.dk], ki));
        }
        let path = x.topology.path(i)?;
        // Forward substitution on the ancestor-masked lower-triangular system.
        for &j in path.iter().take(path.len() - 1) {
            let coefficient = b * ancestor[i * n + j] * dot(ki, &k[j * x.dk..(j + 1) * x.dk]);
            for d in 0..x.dv {
                u[i * x.dv + d] = u[i * x.dv + d] - coefficient * u[j * x.dv + d];
            }
        }
        for d in 0..x.dv {
            outputs[i * x.dv + d] = base[i] * dot(&s0[d * x.dk..(d + 1) * x.dk], qi);
        }
        for j in path {
            let coefficient = ancestor[i * n + j] * dot(qi, &k[j * x.dk..(j + 1) * x.dk]);
            for d in 0..x.dv {
                outputs[i * x.dv + d] = outputs[i * x.dv + d] + coefficient * u[j * x.dv + d];
            }
        }
    }
    Ok(Compact {
        outputs,
        pseudo_values: u,
        ancestor_decay: ancestor,
        base_decay: base,
        initial: s0,
        keys: k,
        topology: x.topology.clone(),
        dk: x.dk,
        dv: x.dv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Operands {
        Operands {
            topology: Topology::new(vec![None, Some(0), Some(0), Some(1), Some(2)]).unwrap(),
            dk: 2,
            dv: 2,
            initial: vec![0.2, -0.5, 0.7, 0.4],
            q: vec![0.8, 0.6, 0.6, -0.8, 0.0, 1.0, 1.0, 0.0, -0.8, 0.6],
            k: vec![0.8, 0.6, 0.6, -0.8, 0.0, 1.0, 1.0, 0.0, -0.8, 0.6],
            v: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8, 0.9, -1.0],
            decay: vec![0.9, 0.8, 0.0, 0.99, 1.0],
            beta: vec![0.5, 1.0, 0.7, 0.0, 0.2],
        }
    }
    fn near(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (&a, &b) in a.iter().zip(b) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }
    #[test]
    fn every_commit_matches_independent_matrix_recurrence() {
        let x = fixture();
        let reference = explicit::<f64>(&x).unwrap();
        let candidate = compact::<f64>(&x).unwrap();
        near(&reference.outputs, &candidate.outputs);
        near(&candidate.reconstruct(None).unwrap(), &x.initial);
        for i in 0..x.topology.len() {
            near(
                &reference.states[i],
                &candidate.reconstruct(Some(i)).unwrap(),
            );
            near(&reference.states[i], &replay::<f64>(&x, Some(i)).unwrap());
        }
    }
    #[test]
    fn sibling_mutation_cannot_contaminate_selected_path() {
        let mut x = fixture();
        let before = compact::<f32>(&x).unwrap();
        x.v[2 * x.dv..3 * x.dv].fill(1000.0);
        x.decay[2] = 1.0;
        let after = compact::<f32>(&x).unwrap();
        for i in [0, 1, 3] {
            assert_eq!(
                &before.outputs[i * x.dv..(i + 1) * x.dv],
                &after.outputs[i * x.dv..(i + 1) * x.dv]
            );
        }
        assert_eq!(
            before.reconstruct(Some(3)).unwrap(),
            after.reconstruct(Some(3)).unwrap()
        );
        assert_ne!(
            before.reconstruct(Some(4)).unwrap(),
            after.reconstruct(Some(4)).unwrap()
        );
        // Fault sensitivity: flattening siblings into a chain must be detectable.
        x.topology = Topology::new(vec![None, Some(0), Some(1), Some(2), Some(3)]).unwrap();
        assert_ne!(
            after.reconstruct(Some(3)).unwrap(),
            compact::<f32>(&x).unwrap().reconstruct(Some(3)).unwrap()
        );
    }
    #[test]
    fn zero_and_underflowed_prefix_decay_stay_finite() {
        let n: usize = 256;
        let x = Operands {
            topology: Topology::new((0..n).map(|i| i.checked_sub(1)).collect()).unwrap(),
            dk: 1,
            dv: 1,
            initial: vec![1.0],
            q: vec![1.0; n],
            k: vec![1.0; n],
            v: vec![0.25; n],
            decay: vec![0.01; n],
            beta: vec![0.5; n],
        };
        let c = compact::<f32>(&x).unwrap();
        let r = explicit::<f32>(&x).unwrap();
        assert!(c.outputs.iter().all(|x| x.is_finite()));
        for (&a, &b) in c.outputs.iter().zip(&r.outputs) {
            assert!((a - b).abs() < 1e-6);
        }
    }
    #[test]
    fn malformed_parent_shapes_and_nonfinite_inputs_are_rejected() {
        assert!(Topology::new(vec![Some(0)]).is_err());
        assert!(Topology::new(vec![None, Some(2), Some(1)]).is_err());
        let mut x = fixture();
        x.q.pop();
        assert!(compact::<f32>(&x).is_err());
        let mut x = fixture();
        x.initial[0] = f64::NAN;
        assert!(compact::<f64>(&x).is_err());
        assert!(fixture().topology.path(5).is_err());
    }
}
