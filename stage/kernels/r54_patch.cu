// R54 G1 (Opus, direct, 2026-11-06): recorded-graph parameter patching probe. Each launch
// writes its own three parameters into its slot; a replay after patching must show the
// patched values for patched launches and the previous values for the others.
extern "C" __global__ void r54_echo(unsigned* out, unsigned idx, unsigned a, unsigned b) {
  if (threadIdx.x == 0 && blockIdx.x == 0) { out[idx * 3 + 0] = idx; out[idx * 3 + 1] = a; out[idx * 3 + 2] = b; }
}
