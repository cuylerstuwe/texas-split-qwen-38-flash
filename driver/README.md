# driver — libtinynv with texas-split's patches

`libtinynv/` is the userspace NVIDIA driver from [macuda](https://github.com/Davinchy/macuda)
(`cuda-shim/libtinynv` at commit `e797e313654ad1c7dd86d60667751ab3872c3a12`, MIT, see `LICENSE`): a C port of
tinygrad's userspace NVIDIA driver that boots the GPU's GSP firmware, manages its page tables and submits
work through hardware queues, reaching the card through tinygrad's TinyGPU DriverKit extension. The stage
server links it statically.

The `*.patch` files are the complete difference from macuda's tree, applied in this order. Every addition is
off unless a stage feature asks for it; with the knobs unset, recording and replay are macuda's.

| Patch | Adds |
|---|---|
| `R38-PDL.patch` | `TINYNV_R38_PDL`: Blackwell QMD V05 launch-latch fields between chained descriptors of recorded graphs |
| `R39-SELECTIVE-PDL.patch` | early launch gated per successor kernel (name prefix `r39pdl_`) |
| `R44-SIDE.patch` | a bounded side-launch mode for one exact five-node graph shape (diagnostic) |
| `R45-CAPTURE-MODE.patch` | `tinynv_r45_capture_mode(stream, mode)`, outside an active capture |
| `R53-CALIBRATION.patch` | `tinynv_r53_stamp(stream, va, slots)` completion stamps; `TINYNV_R53_SMEM_FLOOR_KB` |
| `R53-TRIGGER.patch` | an early-trigger bit for recorded launches (128) |
| `R53-HOST.patch` | host submission knobs (`TINYNV_R53_RING_PIPELINE`, `TINYNV_R53_SKIP_DONE_ACQUIRE`, ...) |
| `R54-GRAPH-PATCH.patch` | `tinynv_r54_graph_patch(stream, graph, launch, offset, data, len)`: re-target a sealed recorded graph's launch parameters for later replays (the stage server re-targets one captured graph per row count every forward) |
| `R65-SIDE-EDGES.patch` | `tinynv_r65_edge(stream, arrive, wait, early)` inside a capture: the next recorded launch arrives at a latch on completion, waits for a latch before it starts (`WAIT_ON_LATCH` + L1/texture invalidates) and/or lets its predecessor release it at its last CTA launch (`PRE_EXIT_AT_LAST_CTA_LAUNCH`). A latch only holds a kernel that executes `griddepcontrol.wait`. Used by the stage's shared-expert side branches (`QD_STAGE_SIDE=1`) |

Build (what `install.sh driver` does):

```sh
sh libtinynv/tools/fetch_nv_headers.sh      # NVIDIA open-gpu-kernel-modules headers, pinned
sh libtinynv/tools/fetch_firmware.sh        # GB202 GSP firmware 570.144 from linux-firmware, pinned
make -C libtinynv BUILD=$PWD/build CC="cc -isysroot $(sh build/sdk.sh)" BUILD_ID=e797e31-r65side $PWD/build/libtinynv.a
```

The firmware directory is compiled in (`TINYNV_FW_DEFAULT_DIR`); `TINYNV_FW_DIR` overrides it.
