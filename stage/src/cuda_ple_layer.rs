//! Complete layer1: PLE -> HC/GDN/HC/MoE/residual, one CUDA queue and one
//! logical accepted-prefix owner for both token/PLE and recurrent state.
use crate::{
    cuda::{Buffer, Device, Graph, Module, PreparedCall},
    cuda_lowbit_layer::{self, Modules},
    cuda_ple::{self, Ple},
    gdn, lowbit_layer,
    mcg_moe::Bank,
    ple,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{cell::Cell, time::Instant};
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Policy {
    pub body: cuda_lowbit_layer::Policy,
    pub ple_fused: bool,
}
impl Policy {
    fn ple(self) -> cuda_ple::Policy {
        cuda_ple::Policy {
            fused: self.ple_fused,
            projection: if self.body.wide { 3 } else { 0 },
        }
    }
}
pub struct Layer<'m, 'd> {
    d: &'d Device,
    ple: Ple<'m, 'd>,
    body: cuda_lowbit_layer::Layer<'m, 'd>,
    rows: Cell<usize>,
    epoch: Cell<u64>,
    ready: Cell<bool>,
    valid: Cell<Option<(u64, bool)>>,
}
impl<'m, 'd> Layer<'m, 'd> {
    pub fn new(
        d: &'d Device,
        m: &'m Modules<'d>,
        ops: &'m Module<'d>,
        w: &lowbit_layer::Weights,
        pw: &ple::Weights,
        bank: &Bank,
    ) -> Result<Self> {
        Self::new_with_prefix_capacity(d, m, ops, w, pw, bank, gdn::MAX_T)
    }
    pub fn new_with_prefix_capacity(
        d: &'d Device,
        m: &'m Modules<'d>,
        ops: &'m Module<'d>,
        w: &lowbit_layer::Weights,
        pw: &ple::Weights,
        bank: &Bank,
        prefix_capacity: usize,
    ) -> Result<Self> {
        ensure!(w.layer == 1, "complete PLE layer only");
        ensure!(
            (1..=gdn::MAX_T).contains(&prefix_capacity),
            "PLE layer GDN prefix capacity"
        );
        Ok(Self {
            d,
            ple: Ple::new(d, &m.affine, ops, pw)?,
            body: cuda_lowbit_layer::Layer::after_ple(d, m, w, bank, prefix_capacity)?,
            rows: Cell::new(0),
            epoch: Cell::new(0),
            ready: Cell::new(false),
            valid: Cell::new(None),
        })
    }
    pub fn description(&self) -> Value {
        json!({"layer":1,"ple":self.ple.description(),"body":self.body.description(),
        "host_intermediate_handoffs":0,"state_owner":"joint token/PLE convolution/GDN prefix; any mutation failure invalidates wrapper",
        "scope":"one complete PLE recurrent layer; not a complete model"})
    }
    fn invalidate(&self) {
        self.rows.set(0);
        self.valid.set(None);
        self.epoch.set(self.epoch.get().wrapping_add(1));
        self.ple.invalidate_output();
        self.body.invalidate_output();
    }
    pub fn prepare(
        &self,
        x: &[u16],
        previous: [u32; 2],
        conv: &[u16],
        initial: &gdn::Initial,
        tokens: &[u32],
        packed: &ple::PackedRows,
    ) -> Result<()> {
        self.prepare_state(previous, conv, initial)?;
        self.prepare_input(x, tokens, packed)
    }
    pub(crate) fn prepare_state(
        &self,
        previous: [u32; 2],
        conv: &[u16],
        initial: &gdn::Initial,
    ) -> Result<()> {
        self.invalidate();
        self.ready.set(false);
        self.ple.initialize(previous, conv)?;
        self.body.prepare_initial(initial)?;
        self.ready.set(true);
        Ok(())
    }
    /// R58 seeding: prepare_state with the body's lean (canary-free) initial upload.
    pub(crate) fn prepare_state_lean(&self, previous: [u32; 2], conv: &[u16], initial: &gdn::Initial) -> Result<()> {
        self.invalidate();
        self.ready.set(false);
        self.ple.initialize(previous, conv)?;
        self.body.prepare_initial_lean(initial)?;
        self.ready.set(true);
        Ok(())
    }
    pub fn prepare_input(&self, x: &[u16], tokens: &[u32], packed: &ple::PackedRows) -> Result<()> {
        self.invalidate();
        ensure!(self.ready.get(), "PLE layer needs coherent initial state");
        self.ple.prepare(tokens, packed, x)?;
        self.body.prepare_resident(tokens.len())?;
        self.rows.set(tokens.len());
        Ok(())
    }
    pub fn initial(&self) -> Result<([u32; 2], Vec<u16>, gdn::Initial)> {
        ensure!(self.ready.get(), "PLE layer state unavailable");
        let (tokens, conv) = self.ple.state()?;
        Ok((tokens, conv, self.body.initial()?))
    }
    pub(crate) fn prepare_resident(&self, tokens: &[u32], packed: &ple::PackedRows) -> Result<()> {
        self.invalidate();
        ensure!(self.ready.get(), "PLE layer needs coherent initial state");
        self.ple.prepare_tokens(tokens, packed)?;
        self.body.prepare_resident(tokens.len())?;
        self.rows.set(tokens.len());
        Ok(())
    }
    pub(crate) fn prepare_resident_lean(&self, tokens: &[u32], packed: &ple::PackedRows) -> Result<()> {
        self.invalidate();
        ensure!(self.ready.get(), "PLE layer needs coherent initial state");
        self.ple.prepare_tokens_lean(tokens, packed)?;
        self.body.prepare_resident(tokens.len())?;
        self.rows.set(tokens.len());
        Ok(())
    }
    pub(crate) fn first_projection_regions(&self) -> Vec<(u64, usize)> {
        self.body.first_projection_regions()
    }
    pub(crate) fn set_mlp_pump(&self, regions: &[(u64, usize)]) -> Result<()> {
        self.body.set_mlp_pump(regions)
    }
    /// R58 sibling screen: this layer's last routed expert ids.
    pub(crate) fn route_ids(&self, rows: usize) -> Result<Vec<u32>> {
        self.body.route_ids(rows)
    }
    /// R58 S3: zero the body's MLP output (before skipping its MLP).
    pub(crate) fn zero_mlp_output(&self) -> Result<()> {
        self.body.zero_mlp_output()
    }
    pub(crate) fn output(&self) -> &Buffer<'d> {
        self.body.output()
    }
    fn record(&self, p: Policy) -> Result<Vec<Vec<PreparedCall<'_, 'm, 'd>>>> {
        self.record_from(self.ple.input(), p)
    }
    pub(crate) fn record_from<'a>(
        &'a self,
        input: &'a Buffer<'d>,
        p: Policy,
    ) -> Result<Vec<Vec<PreparedCall<'a, 'm, 'd>>>> {
        ensure!(
            self.rows.get() > 0 && self.ready.get(),
            "PLE layer record readiness"
        );
        let mut c = vec![self.ple.record_from(input, self.rows.get(), p.ple())?];
        c.extend(
            self.body
                .record_from(self.ple.output(), self.rows.get(), p.body)?,
        );
        Ok(c)
    }
    pub(crate) fn poison(&self, p: Policy) -> Result<()> {
        self.valid.set(None);
        self.ple.invalidate_output();
        self.ple.poison()?;
        self.body.poison(p.body)
    }
    pub(crate) fn collect(&self, p: Policy) -> Result<Run> {
        ensure!(
            self.ready.get() && self.rows.get() > 0,
            "PLE layer collect readiness"
        );
        let ple = self.ple.collect()?;
        let body = self
            .body
            .collect_from(self.ple.output(), self.rows.get(), p.body)?;
        ensure!(
            ple.core.output == body.pre.stream,
            "PLE output reaches recurrent body exactly"
        );
        self.valid.set(Some((self.epoch.get(), p.body.all_states)));
        Ok(Run {
            ple,
            body,
            total_us: 0.,
            enqueue_us: 0.,
            wait_us: 0.,
            phase_us: vec![],
        })
    }
    pub fn plan(&self, p: Policy, repetitions: usize, graph: bool) -> Result<Plan<'_, 'm, 'd>> {
        ensure!((1..=2).contains(&repetitions), "PLE layer repeat bound");
        let c = self.record(p)?;
        let calls = c.iter().map(Vec::len).sum::<usize>();
        ensure!(calls * repetitions <= 128, "PLE layer graph budget");
        let exec = if graph {
            let mut c = c.into_iter().flatten().collect::<Vec<_>>();
            for _ in 1..repetitions {
                c.extend(self.record(p)?.into_iter().flatten());
            }
            Executor::Graph(self.d.capture(c)?)
        } else {
            Executor::Groups(c)
        };
        Ok(Plan {
            owner: self,
            epoch: self.epoch.get(),
            rows: self.rows.get(),
            p,
            exec,
            repetitions,
            calls,
        })
    }
    /// R56 lean drafting commit: the body's then the PLE's commit calls.
    pub(crate) fn record_commit(&self, accepted: usize, p: Policy) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        let mut v = self.body.record_commit(accepted, p.body)?;
        v.extend(self.ple.record_commit(accepted)?);
        Ok(v)
    }
    /// R58 batched prefill, zero-copy: only the PLE's own commit calls (the body's GDN state is
    /// updated in place; the model copies its convolution history).
    pub(crate) fn record_commit_ple(&self, accepted: usize) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        self.ple.record_commit(accepted)
    }
    pub(crate) fn body_gdn(&self) -> &crate::cuda_gdn::Gdn<'m, 'd> {
        self.body.gdn_ref()
    }
    pub(crate) fn state_buffers(&self) -> Vec<&Buffer<'d>> {
        let mut v = self.body.state_buffers();
        v.extend(self.ple.state_buffers());
        v
    }
    pub(crate) fn previous_tokens(&self) -> [u32; 2] {
        self.ple.previous_tokens()
    }
    pub(crate) fn restore_meta(&self, previous: [u32; 2]) {
        self.rows.set(0);
        self.valid.set(None);
        self.epoch.set(self.epoch.get().wrapping_add(1));
        self.body.invalidate_pending();
        self.ple.set_previous_tokens(previous);
        self.ready.set(true);
    }
    pub(crate) fn finish_commit(&self, accepted: usize) -> Result<()> {
        self.rows.set(0);
        self.valid.set(None);
        self.epoch.set(self.epoch.get().wrapping_add(1));
        self.body.finish_commit();
        self.ple.finish_commit(accepted)?;
        self.ready.set(true);
        Ok(())
    }
    pub fn commit(&self, accepted: usize) -> Result<()> {
        let (epoch, all) = self
            .valid
            .get()
            .ok_or_else(|| anyhow::anyhow!("PLE layer commit before completed full layer"))?;
        let rows = self.rows.get();
        ensure!(
            epoch == self.epoch.get()
                && accepted <= rows
                && (all || accepted == 0 || accepted == rows),
            "PLE layer unavailable prefix"
        );
        // Child completion tokens are needed for their commits. Invalidate only
        // this wrapper first; no caller can use a partially mutated pair.
        self.rows.set(0);
        self.valid.set(None);
        self.epoch.set(self.epoch.get().wrapping_add(1));
        self.ready.set(false);
        self.body.commit(accepted)?;
        self.ple.commit(accepted)?;
        self.ready.set(true);
        Ok(())
    }
}
enum Executor<'a, 'm, 'd> {
    Graph(Graph<'a, 'm, 'd>),
    Groups(Vec<Vec<PreparedCall<'a, 'm, 'd>>>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Layer<'m, 'd>,
    epoch: u64,
    rows: usize,
    p: Policy,
    exec: Executor<'a, 'm, 'd>,
    repetitions: usize,
    pub calls: usize,
}
pub struct Run {
    pub ple: cuda_ple::Run,
    pub body: cuda_lowbit_layer::Run,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub wait_us: f64,
    pub phase_us: Vec<f64>,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self, wait_phase: bool) -> Result<Run> {
        let a = self.owner;
        ensure!(
            a.ready.get() && a.epoch.get() == self.epoch && a.rows.get() == self.rows,
            "PLE layer stale plan"
        );
        ensure!(
            !wait_phase || matches!(self.exec, Executor::Groups(_)),
            "PLE layer phase waits require prepared groups"
        );
        a.poison(self.p)?;
        a.d.sync()?;
        let start = Instant::now();
        let (mut enqueue_us, mut wait_us) = (0., 0.);
        let mut phase_us = vec![];
        match &self.exec {
            Executor::Graph(g) => {
                let t = Instant::now();
                g.launch()?;
                enqueue_us = t.elapsed().as_secs_f64() * 1e6;
            }
            Executor::Groups(c) => {
                for _ in 0..self.repetitions {
                    for group in c {
                        let phase = Instant::now();
                        let t = Instant::now();
                        for k in group {
                            k.enqueue()?;
                        }
                        enqueue_us += t.elapsed().as_secs_f64() * 1e6;
                        if wait_phase {
                            let t = Instant::now();
                            a.d.sync()?;
                            wait_us += t.elapsed().as_secs_f64() * 1e6;
                            phase_us.push(phase.elapsed().as_secs_f64() * 1e6);
                        }
                    }
                }
            }
        }
        let t = Instant::now();
        a.d.sync()?;
        wait_us += t.elapsed().as_secs_f64() * 1e6;
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        let mut out = a.collect(self.p)?;
        out.total_us = total_us;
        out.enqueue_us = enqueue_us;
        out.wait_us = wait_us;
        out.phase_us = phase_us;
        Ok(out)
    }
}
