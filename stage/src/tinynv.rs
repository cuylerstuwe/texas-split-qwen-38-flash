//! Raw FFI for the pinned MIT libtinynv ABI. These functions are deliberately
//! unsafe: the caller owns GPU/host lifetimes, bounds, ordering and completion.
//! Header pin and license are retained in evidence pins and NOTICE.txt.
use std::ffi::{c_char, c_int, c_uint, c_void};

pub type Handle = *mut c_void;
pub type DevPtr = u64;

#[repr(C)]
pub struct DeviceProps {
    pub name: [c_char; 256],
    pub cc_major: c_int,
    pub cc_minor: c_int,
    pub sm_count: c_int,
    pub warp_size: c_int,
    pub total_mem: usize,
    pub max_threads_per_block: c_int,
    pub max_shared_per_block: c_int,
    pub max_shared_per_sm: c_int,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Param {
    pub offset: c_int,
    pub size: c_int,
}
#[repr(C)]
pub struct KernelInfo {
    pub num_params: c_int,
    pub param_base: c_int,
    pub params: [Param; 64],
    pub regs: c_int,
    pub static_smem: c_int,
    pub max_threads: c_int,
    pub attr_reported: c_uint,
    pub const_size_bytes: c_int,
    pub local_size_bytes: c_int,
    pub ptx_version: c_int,
    pub cache_mode_ca: c_int,
    pub cluster_size_must_be_set: c_int,
    pub required_cluster: [c_int; 3],
}

unsafe extern "C" {
    pub fn tinynv_last_error() -> *const c_char;
    pub fn tinynv_build_id() -> *const c_char;
    pub fn tinynv_init() -> c_int;
    pub fn tinynv_device_count() -> c_int;
    pub fn tinynv_device_get(out: *mut Handle, ordinal: c_int) -> c_int;
    pub fn tinynv_device_props(device: Handle, out: *mut DeviceProps) -> c_int;
    pub fn tinynv_module_load(
        device: Handle,
        cubin: *const c_void,
        len: usize,
        out: *mut Handle,
    ) -> c_int;
    pub fn tinynv_module_unload(module: Handle) -> c_int;
    pub fn tinynv_get_kernel(module: Handle, name: *const c_char, out: *mut Handle) -> c_int;
    pub fn tinynv_kernel_info(kernel: Handle, out: *mut KernelInfo) -> c_int;
    pub fn tinynv_malloc(device: Handle, bytes: usize, out: *mut DevPtr) -> c_int;
    pub fn tinynv_free(device: Handle, ptr: DevPtr) -> c_int;
    pub fn tinynv_mem_info(device: Handle, free_bytes: *mut u64, total_bytes: *mut u64) -> c_int;
    pub fn tinynv_memcpy_htod(
        stream: Handle,
        dst: DevPtr,
        src: *const c_void,
        bytes: usize,
    ) -> c_int;
    pub fn tinynv_memcpy_dtoh(stream: Handle, dst: *mut c_void, src: DevPtr, bytes: usize)
    -> c_int;
    pub fn tinynv_stream_create(device: Handle, out: *mut Handle) -> c_int;
    pub fn tinynv_stream_destroy(stream: Handle) -> c_int;
    pub fn tinynv_stream_sync(stream: Handle) -> c_int;
    // R58: non-blocking completion of work recorded on a stream (driver-r38 include/tinynv.h).
    pub fn tinynv_stream_flush(stream: Handle) -> c_int;
    pub fn tinynv_event_create(device: Handle, out: *mut Handle) -> c_int;
    pub fn tinynv_event_record(event: Handle, stream: Handle) -> c_int;
    pub fn tinynv_event_query(event: Handle, complete: *mut c_int) -> c_int;
    pub fn tinynv_event_destroy(event: Handle) -> c_int;
    pub fn tinynv_set_download_kernel(
        device: Handle,
        kernel: Handle,
        threads: c_uint,
        bytes_per_thread: c_uint,
    ) -> c_int;
    pub fn tinynv_graph_begin(stream: Handle) -> c_int;
    #[cfg(feature = "r45-driver")]
    pub fn tinynv_r45_capture_mode(stream: Handle, mode: c_int) -> c_int;
    #[cfg(feature = "r53-driver")]
    pub fn tinynv_r53_stamp(stream: Handle, va: u64, slots: c_uint) -> c_int;
    #[cfg(feature = "r53-driver")]
    pub fn tinynv_r54_graph_patch(stream: Handle, graph: Handle, launch: c_uint, offset: c_uint,
        data: *const std::ffi::c_void, len: c_uint) -> c_int;
    pub fn tinynv_graph_end(stream: Handle, out: *mut Handle) -> c_int;
    pub fn tinynv_graph_launch(stream: Handle, graph: Handle) -> c_int;
    pub fn tinynv_graph_free(stream: Handle, graph: Handle);
    pub fn tinynv_launch(
        stream: Handle,
        kernel: Handle,
        gx: c_uint,
        gy: c_uint,
        gz: c_uint,
        bx: c_uint,
        by: c_uint,
        bz: c_uint,
        smem: c_uint,
        params: *const c_void,
        len: usize,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;
    // Checked independently by xcrun clang against the pinned actual C header:
    // evidence/r02-abi/{abi.c,c-layout.json}. No GPU is opened by this test.
    #[test]
    fn c_abi_layout_matches_header_probe() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<DeviceProps>(), 296);
        assert_eq!(offset_of!(DeviceProps, total_mem), 272);
        assert_eq!(size_of::<KernelInfo>(), 568);
        assert_eq!(offset_of!(KernelInfo, params), 8);
        assert_eq!(offset_of!(KernelInfo, max_threads), 528);
    }
}
