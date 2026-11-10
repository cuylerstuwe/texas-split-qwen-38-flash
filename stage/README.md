# stage — the RTX 5090 pipeline stage

`canon-stage-server` holds transformer layers **28–47** of the Sushi Qwen3.8-Flash-Next 4bpw pack on an
RTX 5090 (5 full-attention layers with 8-bit affine KV and QSA block selection, 15 gated-delta-net layers,
their EXL3 4-bit MoE experts) and serves them to Sushi over a Unix socket. The protocol is documented at
the top of `src/bin/canon_stage_server.rs`: FORWARD (decode/verify rows, with deferred partial-accept
COMMIT), SEED (load a sequence's state), PREFILL (pipelined chunks, optional state pull-back), PARK /
RESTORE (a parked background request's whole stage state to and from an SSD file, verified byte for byte
when `QD_STAGE_RESTORE_VERIFY=1`), INFO, QUIT.

This crate is the stage's part of a larger research codebase; only the stage server binary is built.
CUDA kernels ship as prebuilt `kernels/*.cubin` (sources `kernels/*.cu`), each pinned by SHA-256
(`pins/*.json`, constants in the source) and verified when loaded, so no CUDA toolkit is needed.

Build (what `install.sh stage` does; needs `../driver/build/libtinynv.a`):

```sh
QWEN_TINYNV_ARCHIVE=$PWD/../driver/build/libtinynv.a cargo build --release --features cuda,metal,r65-driver --bin canon-stage-server
```

`r65-driver` needs the driver's `tinynv_r65_edge` (`../driver/R65-SIDE-EDGES.patch`): with `QD_STAGE_SIDE=1` each MoE
block's shared expert runs as a side branch beside the router chain and the routed experts (QMD launch latches;
the joins are `kernels/r65_wait.cu`'s copies of three kernels with `griddepcontrol.wait` at entry). Outputs are
bit-identical; 1–7 row forwards are 0.24–0.36 ms faster. Without the switch the recording is unchanged.

Run (the host does this): `canon-stage-server FIRST CAPACITY` (28, 262160) under macuda's
`tools/nv_shim_step.sh local probe` (read-only GPU preflight, TinyGPU server, logged step), with
`QWEN_HYBRID_GUARDED=1`, `QD_STAGE_SOCKET`, `QD_STAGE_SHM`, `QWEN_HYBRID_MODEL` (the model directory),
`TINYNV_FW_DIR` and the `TINYNV_*` settings in `host/texas_split.py` (`launch_stage`). At 262,160 rows the
stage leaves ~0.8 GiB of the card free. It exits when its client disconnects or on QUIT; on SIGTERM the
driver puts the card down cleanly — never SIGKILL it.
