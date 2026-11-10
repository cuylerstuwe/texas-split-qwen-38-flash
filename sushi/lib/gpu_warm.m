// R65: keep the Mac GPU's clocks up while the decode thread waits for the 5090 stage (native/stage_split.cpp,
// SUSHI_R65_GPU_WARM). An idle Apple GPU clocks down within a few ms: on this M5 Max the same work after a 3-12 ms
// idle gap ran ~40% slower than back to back, and in a decode-round-shaped microbenchmark (14 ms of work, a 6.5 ms
// gap, then the timed post-reply work) one 256-thread kernel every 0.5 ms during the gap cut the post-reply work by
// 23% and the next long stretch by 1-4%. The tick touches only its own 1 KB private buffer, on a queue of its own.
#import <Metal/Metal.h>

static id<MTLCommandQueue> g_q;
static id<MTLComputePipelineState> g_pso;
static id<MTLBuffer> g_buf;
static int g_state;  // 0 untried, 1 ready, -1 failed

int gpu_warm_init(void) {
    if (g_state) return g_state > 0 ? 0 : -1;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = dev ? [dev newLibraryWithSource:
            @"#include <metal_stdlib>\nusing namespace metal;\n"
             "kernel void r65_warm_tick(device float *a [[buffer(0)]], uint i [[thread_position_in_grid]]) { a[i] += 1.0f; }\n"
                                                      options:nil error:&err] : nil;
        id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"r65_warm_tick"] : nil;
        g_pso = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
        g_q = dev ? [dev newCommandQueue] : nil;
        g_buf = dev ? [dev newBufferWithLength:256 * sizeof(float) options:MTLResourceStorageModePrivate] : nil;
        g_state = (g_pso && g_q && g_buf) ? 1 : -1;
    }
    return g_state > 0 ? 0 : -1;
}

void gpu_warm_tick(void) {
    if (g_state <= 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g_pso];
        [e setBuffer:g_buf offset:0 atIndex:0];
        [e dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [cb commit];
    }
}
