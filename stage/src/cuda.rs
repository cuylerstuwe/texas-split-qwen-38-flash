//! Small single-owner native CUDA backend over pinned libtinynv. No host model
//! runtime. Buffers/modules borrow the device; drop drains before freeing.
//! Kernel launch remains unsafe because a cubin defines its own memory accesses.
use crate::tinynv::*;
use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    path::Path,
    ptr::null_mut,
    rc::Rc,
};

const COPY: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/macuda/copy1d.sm_120.cubin");
const COPY_SHA: &str = "03a644f4a55d702f0aa2bf969fa0e716fcf89e2d948f42f0bbcddc7e0ede4c58";
fn check(s: i32) -> Result<()> {
    if s != 0 {
        let message = unsafe { CStr::from_ptr(tinynv_last_error()) }.to_string_lossy();
        anyhow::bail!("libtinynv {s}: {message}");
    }
    Ok(())
}
fn pinned(path: &Path, expected: &str) -> Result<Vec<u8>> {
    let data = std::fs::read(path)?;
    ensure!(
        format!("{:x}", Sha256::digest(&data)) == expected,
        "cubin hash changed: {}",
        path.display()
    );
    Ok(data)
}
pub struct Device {
    handle: Handle,
    stream: Handle,
    copy_module: Handle,
    _copy_bytes: Vec<u8>,
    _single_owner: PhantomData<Rc<()>>,
    arena: std::cell::RefCell<Arena>,
    /// R58: the completion marker for asynchronous drafter steps (created on first use).
    marker: std::cell::Cell<Handle>,
}
/// R58 VRAM arena. The driver charges every allocation 64 KiB of slack past its end plus page
/// rounding (r54-alloc-audit-02: ~185 small buffers per recurrent layer -> +12 MiB of a 16.5 MiB
/// overhead). While an arena scope is open, allocations that the driver would place on 4 KiB
/// pages (under 8 MiB with its slack) are carved out of 64 MiB chunks instead, page-aligned with a
/// 4 KiB gap after each (QWEN_R58_ARENA_PAD), so reads just past an end still land in mapped
/// memory. Sub-buffers are never freed one by one (a bump allocator: open scopes only around
/// long-lived allocations); the chunks are freed when the device closes. QWEN_R58_ARENA=0 = off.
#[derive(Default)]
struct Arena {
    depth: usize,
    chunks: Vec<DevPtr>,
    cur: DevPtr,
    end: DevPtr,
    buffers: usize,
    requested: usize,
}
const ARENA_CHUNK: usize = 64 << 20;
const ARENA_ALIGN: u64 = 4096;
/// The driver's own slack per allocation (TINYNV_ALLOC_SLACK_KB default 64).
const DRIVER_SLACK: usize = 64 << 10;
/// Allocations from here on get 2 MiB pages from the driver; the arena leaves those alone.
const DRIVER_BIG: usize = 8 << 20;
/// An open arena scope (see Device::arena_scope).
pub struct ArenaScope<'d> {
    d: &'d Device,
}
impl Drop for ArenaScope<'_> {
    fn drop(&mut self) {
        self.d.arena.borrow_mut().depth -= 1;
    }
}
/// Allocator-managed VRAM, not the board's physical capacity. Free space alone
/// does not guarantee that a particular contiguous allocation can succeed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct MemoryInfo {
    pub free_bytes: u64,
    pub managed_bytes: u64,
}
impl Device {
    pub fn open() -> Result<Self> {
        for (key, value) in [
            ("QWEN_HYBRID_GUARDED", "1"),
            ("TINYNV_HW", "1"),
            ("TINYNV_INLINE_UPLOAD", "0"),
            ("TINYNV_DOWNLOAD_VIA_COMPUTE", "1"),
        ] {
            ensure!(
                std::env::var(key).as_deref() == Ok(value),
                "this backend requires {key}={value}"
            );
        }
        let mut this = Self {
            handle: null_mut(),
            stream: null_mut(),
            copy_module: null_mut(),
            _copy_bytes: pinned(Path::new(COPY), COPY_SHA)?,
            _single_owner: PhantomData,
            arena: Default::default(),
            marker: std::cell::Cell::new(null_mut()),
        };
        unsafe {
            check(tinynv_init())?;
            ensure!(tinynv_device_count() == 1, "one physical device required");
            check(tinynv_device_get(&mut this.handle, 0))?;
            ensure!(!this.handle.is_null(), "null CUDA device");
            let mut p: DeviceProps = std::mem::zeroed();
            check(tinynv_device_props(this.handle, &mut p))?;
            let name: String = p
                .name
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8 as char)
                .collect();
            ensure!(
                (name.starts_with("GB202") || name.contains("5090"))
                    && p.cc_major == 12
                    && p.cc_minor == 0
                    && p.total_mem >= 24 * 1024usize.pow(3),
                "unexpected CUDA device"
            );
            check(tinynv_stream_create(this.handle, &mut this.stream))?;
            check(tinynv_module_load(
                this.handle,
                this._copy_bytes.as_ptr().cast(),
                this._copy_bytes.len(),
                &mut this.copy_module,
            ))?;
            let mut copy = null_mut();
            check(tinynv_get_kernel(
                this.copy_module,
                c"tinycudart_copy1d".as_ptr(),
                &mut copy,
            ))?;
            check(tinynv_set_download_kernel(this.handle, copy, 256, 16))?;
            println!(
                "{}",
                json!({"kind":"device","real_cuda":true,"name":name,"cc":[p.cc_major,p.cc_minor],"driver_reported_sm_count":p.sm_count,"total_mem":p.total_mem,"driver_build":CStr::from_ptr(tinynv_build_id()).to_string_lossy()})
            );
        }
        Ok(this)
    }
    pub fn sync(&self) -> Result<()> {
        unsafe { check(tinynv_stream_sync(self.stream)) }
    }
    /// R58: record a completion marker after everything enqueued so far and hand the batch to
    /// the card without waiting; `marker_done` then answers whether that work has retired.
    pub fn mark(&self) -> Result<()> {
        let mut ev = self.marker.get();
        if ev.is_null() {
            unsafe { check(tinynv_event_create(self.handle, &mut ev))? };
            self.marker.set(ev);
        }
        unsafe {
            check(tinynv_event_record(ev, self.stream))?;
            check(tinynv_stream_flush(self.stream))
        }
    }
    /// R58: has the work before the last `mark` retired? (never true early; no wait)
    pub fn marker_done(&self) -> Result<bool> {
        let ev = self.marker.get();
        if ev.is_null() {
            return Ok(true);
        }
        let mut complete: std::ffi::c_int = 0;
        unsafe { check(tinynv_event_query(ev, &mut complete))? };
        Ok(complete != 0)
    }
    /// R53 calibration (isolated r53nv archive only): while a graph is recorded,
    /// every descriptor stamps its completion clock into `buf` (16 bytes each,
    /// index < slots) and each replay stamps chain starts after them. Outside
    /// capture only; sealed graphs keep what they were recorded with.
    #[cfg(feature = "r53-driver")]
    pub fn r53_stamp(&self, buf: Option<&Buffer<'_>>, slots: u32) -> Result<()> {
        ensure!(
            option_env!("QWEN_TINYNV_ARCHIVE")
                .unwrap_or("")
                .ends_with("driver-r38/build/r53nv/libtinynv.a")
                || option_env!("QWEN_TINYNV_ARCHIVE").unwrap_or("").ends_with("driver-r38/build/r65nv/libtinynv.a"),
            "R53 isolated archive required"
        );
        let va = match buf {
            Some(b) => {
                ensure!(std::ptr::eq(b.device, self), "foreign stamp buffer");
                ensure!(b.bytes.get() >= 16 * (slots as usize + 128), "stamp region too small");
                b.ptr.get()
            }
            None => 0,
        };
        unsafe { check(tinynv_r53_stamp(self.stream, va, if buf.is_some() { slots } else { 0 })) }
    }
    /// R45 probe-only recording switch; existing sealed graphs are unchanged.
    /// The driver rejects unknown modes and changes during an active capture.
    #[cfg(feature = "r45-driver")]
    pub fn experimental_capture_mode(&self, mode: i32) -> Result<()> {
        ensure!(
            option_env!("QWEN_TINYNV_ARCHIVE")
                .unwrap_or("")
                .ends_with("driver-r38/build/r45nv/libtinynv.a"),
            "R45 isolated archive required"
        );
        self.sync()?;
        unsafe { check(tinynv_r45_capture_mode(self.stream, mode)) }
    }
    /// Probe-only negative control: changing mode in an open empty capture must
    /// fail. End the capture on every path; no GPU kernel is submitted here.
    #[cfg(feature = "r45-driver")]
    pub fn experimental_capture_mode_preflight(&self) -> Result<()> {
        self.experimental_capture_mode(0)?;
        unsafe {
            check(tinynv_graph_begin(self.stream))?;
        }
        let rejected: Vec<_> = [0, 64, 13]
            .into_iter()
            .map(|mode| unsafe { tinynv_r45_capture_mode(self.stream, mode) != 0 })
            .collect();
        let mut graph = null_mut();
        let ended = unsafe { tinynv_graph_end(self.stream, &mut graph) };
        if !graph.is_null() {
            unsafe {
                tinynv_graph_free(self.stream, graph);
            }
        }
        self.experimental_capture_mode(0)?;
        ensure!(
            rejected.iter().all(|v| *v),
            "mode changed during open capture"
        );
        ensure!(ended != 0 && graph.is_null(), "empty capture accepted");
        Ok(())
    }
    /// Queries this owner's existing driver connection; never opens a telemetry
    /// client or submits another GPU workload.
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        ensure!(
            !self.handle.is_null() && !self.stream.is_null(),
            "memory query requires an open CUDA device"
        );
        let mut free_bytes = 0;
        let mut managed_bytes = 0;
        unsafe {
            check(tinynv_mem_info(
                self.handle,
                &mut free_bytes,
                &mut managed_bytes,
            ))?;
        }
        ensure!(
            managed_bytes > 0 && free_bytes <= managed_bytes,
            "CUDA memory report bounds"
        );
        Ok(MemoryInfo {
            free_bytes,
            managed_bytes,
        })
    }
    /// R54: with QWEN_ALLOC_LOG=1, every allocation prints its call site and size
    /// (one JSON line on stdout) for VRAM audits; off by default.
    #[track_caller]
    pub fn alloc(&self, bytes: usize) -> Result<Buffer<'_>> {
        static LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *LOG.get_or_init(|| std::env::var("QWEN_ALLOC_LOG").as_deref() == Ok("1")) {
            let at = std::panic::Location::caller();
            println!("{}", json!({"kind":"alloc","file":at.file(),"line":at.line(),"bytes":bytes}));
        }
        ensure!(
            (1..=1024usize.pow(3)).contains(&bytes),
            "bounded CUDA allocation"
        );
        if let Some(ptr) = self.arena_take(bytes)? {
            return Ok(Buffer { device: self, ptr: std::cell::Cell::new(ptr), bytes: std::cell::Cell::new(bytes), owned: false });
        }
        let buffer = Buffer {
            device: self,
            ptr: std::cell::Cell::new(0),
            bytes: std::cell::Cell::new(bytes),
            owned: true,
        };
        let mut raw: DevPtr = 0;
        unsafe {
            check(tinynv_malloc(self.handle, bytes, &mut raw))?;
        }
        buffer.ptr.set(raw);
        ensure!(
            raw != 0 && raw % 256 == 0,
            "device allocation alignment"
        );
        Ok(buffer)
    }
    /// R58: open an arena scope; allocations under 8 MiB made while it is alive come out of
    /// shared 64 MiB chunks (see Arena). Use only around long-lived allocations.
    pub fn arena_scope(&self) -> ArenaScope<'_> {
        self.arena.borrow_mut().depth += 1;
        ArenaScope { d: self }
    }
    /// R58: arena chunks, the buffers carved from them, the bytes those requested and what the
    /// driver would have charged for them one by one.
    pub fn arena_stats(&self) -> serde_json::Value {
        let a = self.arena.borrow();
        json!({"chunks":a.chunks.len(),"chunk_bytes":ARENA_CHUNK,"buffers":a.buffers,"requested_bytes":a.requested,
               "enabled":arena_enabled(),"pad":arena_pad()})
    }
    fn arena_take(&self, bytes: usize) -> Result<Option<DevPtr>> {
        let mut a = self.arena.borrow_mut();
        if a.depth == 0 || !arena_enabled() || bytes + DRIVER_SLACK >= DRIVER_BIG {
            return Ok(None);
        }
        let round = |v: u64| v.div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
        let need = round(bytes as u64) + arena_pad();
        let mut start = round(a.cur);
        if a.cur == 0 || start + need > a.end {
            // a chunk the driver charges exactly ARENA_CHUNK: its own slack brings it there
            let mut p: DevPtr = 0;
            unsafe {
                check(tinynv_malloc(self.handle, ARENA_CHUNK - DRIVER_SLACK, &mut p))?;
            }
            ensure!(p != 0 && p % ARENA_ALIGN == 0, "arena chunk alignment");
            a.chunks.push(p);
            (a.cur, a.end, start) = (p, p + (ARENA_CHUNK - DRIVER_SLACK) as u64, p);
        }
        ensure!(start + need <= a.end, "arena chunk room");
        a.cur = start + need;
        a.buffers += 1;
        a.requested += bytes;
        Ok(Some(start))
    }
    pub fn module(&self, path: &Path, sha: &str) -> Result<Module<'_>> {
        let mut module = Module {
            device: self,
            handle: null_mut(),
            _bytes: pinned(path, sha)?,
        };
        unsafe {
            check(tinynv_module_load(
                self.handle,
                module._bytes.as_ptr().cast(),
                module._bytes.len(),
                &mut module.handle,
            ))?;
        }
        ensure!(!module.handle.is_null(), "null module");
        Ok(module)
    }
    pub fn close(&mut self) -> Result<()> {
        if !self.stream.is_null() {
            self.sync()?;
            // R58: every sub-buffer borrowed this device, so none is alive here
            for p in std::mem::take(&mut self.arena.get_mut().chunks) {
                unsafe {
                    check(tinynv_free(self.handle, p))?;
                }
            }
            let ev = self.marker.replace(null_mut());
            if !ev.is_null() {
                unsafe { check(tinynv_event_destroy(ev))? };
            }
            unsafe {
                check(tinynv_stream_destroy(self.stream))?;
            }
            self.stream = null_mut();
        }
        if !self.copy_module.is_null() {
            unsafe {
                check(tinynv_module_unload(self.copy_module))?;
            }
            self.copy_module = null_mut();
        }
        Ok(())
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            eprintln!("CUDA device cleanup: {e:#}");
        }
    }
}
impl<'d> Buffer<'d> {
    /// R60 S4 (prefill v2): a non-owning view of bytes [offset, offset + bytes) of this buffer (for
    /// row sub-batches of chunk scratch). The view must not outlive this buffer.
    pub(crate) fn view(&self, offset: usize, bytes: usize) -> Result<Buffer<'d>> {
        ensure!(bytes > 0 && offset + bytes <= self.bytes.get(), "buffer view bounds");
        Ok(Buffer { device: self.device, ptr: std::cell::Cell::new(self.ptr.get() + offset as DevPtr), bytes: std::cell::Cell::new(bytes), owned: false })
    }
    /// R71: re-point this non-owning view to bytes [offset, offset + bytes) of `parent` (a conversation slot's rows
    /// move). Calls recorded or graphs captured before this keep the old address: the caller drops them.
    pub(crate) fn retarget(&self, parent: &Buffer<'d>, offset: usize, bytes: usize) -> Result<()> {
        ensure!(!self.owned && std::ptr::eq(self.device, parent.device), "retarget needs a view of the same device");
        ensure!(bytes > 0 && offset + bytes <= parent.bytes.get(), "buffer retarget bounds");
        self.ptr.set(parent.ptr.get() + offset as DevPtr);
        self.bytes.set(bytes);
        Ok(())
    }
}
pub struct Buffer<'d> {
    device: &'d Device,
    /// R71: in cells so a non-owning view can be re-pointed in place (`retarget`); fixed for every other buffer.
    ptr: std::cell::Cell<DevPtr>,
    bytes: std::cell::Cell<usize>,
    /// R58: false for arena sub-buffers (their chunk is freed with the device).
    owned: bool,
}
fn arena_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R58_ARENA").as_deref() != Ok("0"))
}
fn arena_pad() -> u64 {
    static PAD: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PAD.get_or_init(|| std::env::var("QWEN_R58_ARENA_PAD").ok().and_then(|v| v.parse().ok()).unwrap_or(4096))
}
impl Buffer<'_> {
    /// R56: the raw device address (copy-table entries).
    pub(crate) fn device_ptr(&self) -> u64 {
        self.ptr.get()
    }
    pub(crate) fn byte_len(&self) -> usize {
        self.bytes.get()
    }
    /// R52: the buffer's device virtual address (for pump descriptors passed as U64).
    pub(crate) fn device_addr(&self) -> u64 {
        self.ptr.get()
    }
    pub(crate) fn aliases(&self, other: &Buffer<'_>) -> bool {
        std::ptr::eq(self.device, other.device) && self.ptr.get() == other.ptr.get()
    }
    pub fn upload(&self, data: &[u8]) -> Result<()> {
        ensure!(data.len() <= self.bytes.get(), "upload bounds");
        // Pinned forced-copy path copies caller bytes to staging before return.
        unsafe {
            check(tinynv_memcpy_htod(
                self.device.stream,
                self.ptr.get(),
                data.as_ptr().cast(),
                data.len(),
            ))?;
        }
        self.device.sync()
    }
    /// R59: upload `data` at byte `offset` into this buffer (incremental state refresh).
    pub fn upload_at(&self, offset: usize, data: &[u8]) -> Result<()> {
        ensure!(offset.checked_add(data.len()).is_some_and(|e| e <= self.bytes.get()), "upload_at bounds");
        if data.is_empty() {
            return Ok(());
        }
        unsafe {
            check(tinynv_memcpy_htod(
                self.device.stream,
                self.ptr.get() + offset as u64,
                data.as_ptr().cast(),
                data.len(),
            ))?;
        }
        self.device.sync()
    }
    pub fn download(&self, out: &mut [u8]) -> Result<()> {
        ensure!(out.len() <= self.bytes.get(), "download bounds");
        // Pinned compute download waits and copies to caller storage before return.
        unsafe {
            check(tinynv_memcpy_dtoh(
                self.device.stream,
                out.as_mut_ptr().cast(),
                self.ptr.get(),
                out.len(),
            ))?;
        }
        self.device.sync()
    }
    pub fn close(mut self) -> Result<()> {
        self.release()
    }
    fn release(&mut self) -> Result<()> {
        if self.ptr.get() != 0 && self.owned {
            self.device.sync()?;
            unsafe {
                check(tinynv_free(self.device.handle, self.ptr.get()))?;
            }
        }
        self.ptr.set(0);
        Ok(())
    }
}
impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.release() {
            eprintln!("CUDA buffer cleanup: {e:#}");
        }
    }
}
pub struct Module<'d> {
    device: &'d Device,
    handle: Handle,
    _bytes: Vec<u8>,
}
impl<'d> Module<'d> {
    pub fn kernel(&self, name: &str) -> Result<Kernel<'_, 'd>> {
        let label = name.to_string();
        let name = CString::new(name)?;
        let mut handle = null_mut();
        let mut info = unsafe { std::mem::zeroed() };
        unsafe {
            check(tinynv_get_kernel(self.handle, name.as_ptr(), &mut handle))?;
            check(tinynv_kernel_info(handle, &mut info))?;
        }
        ensure!(!handle.is_null(), "null kernel");
        Ok(Kernel {
            module: self,
            handle,
            info,
            name: label,
        })
    }
    pub fn close(mut self) -> Result<()> {
        self.release()
    }
    fn release(&mut self) -> Result<()> {
        if !self.handle.is_null() {
            self.device.sync()?;
            unsafe {
                check(tinynv_module_unload(self.handle))?;
            }
            self.handle = null_mut();
        }
        Ok(())
    }
}
impl Drop for Module<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.release() {
            eprintln!("CUDA module cleanup: {e:#}");
        }
    }
}
#[derive(Clone, Copy)]
pub enum Arg<'a, 'd> {
    Buffer(&'a Buffer<'d>),
    U32(u32),
    U64(u64),
    F32(f32),
}
pub struct Kernel<'m, 'd> {
    module: &'m Module<'d>,
    handle: Handle,
    info: KernelInfo,
    name: String,
}
impl<'m, 'd> Kernel<'m, 'd> {
    /// The name this kernel was looked up by (R53 calibration labels).
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Registers per thread and static shared bytes, for residency sizing.
    pub fn resources(&self) -> (usize, usize) {
        (
            self.info.regs.max(0) as usize,
            self.info.static_smem.max(0) as usize,
        )
    }
    pub fn description(&self) -> serde_json::Value {
        json!({"parameter_count":self.info.num_params,
            "parameters":self.info.params[..self.info.num_params.max(0).min(64) as usize].iter()
                .map(|p|json!({"offset":p.offset,"bytes":p.size})).collect::<Vec<_>>(),
            "registers":self.info.regs,"static_shared_bytes":self.info.static_smem,
            "max_threads":self.info.max_threads,"local_bytes":self.info.local_size_bytes})
    }
    /// # Safety
    /// Caller must match this kernel's types, geometry, buffer sizes, and alias
    /// restrictions. Buffers/parameters are checked for owner and ABI size only.
    /// No graph capture here: allocation drop synchronizes all queued launches.
    pub unsafe fn launch(
        &self,
        args: &[Arg<'_, '_>],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> Result<()> {
        let blob = self.encode(args, grid, block)?;
        unsafe { self.enqueue_blob(&blob, grid, block) }
    }
    /// # Safety
    /// Same kernel/geometry/buffer contract as launch. The returned record keeps
    /// every argument buffer and the module borrowed until its last use.
    pub unsafe fn prepare<'a>(
        &'a self,
        args: Vec<Arg<'a, 'd>>,
        grid: [u32; 3],
        block: [u32; 3],
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        let blob = self.encode(&args, grid, block)?;
        Ok(PreparedCall {
            kernel: self,
            _args: args,
            blob,
            grid,
            block,
            edge: None,
        })
    }
    fn encode(&self, args: &[Arg<'_, '_>], grid: [u32; 3], block: [u32; 3]) -> Result<Vec<u8>> {
        ensure!(
            args.len() <= 64 && self.info.num_params == args.len() as i32,
            "kernel argument count"
        );
        ensure!(
            grid.iter().all(|&d| d > 0)
                && block.iter().all(|&d| d > 0)
                && block.iter().map(|&d| u64::from(d)).product::<u64>() <= 1024,
            "launch geometry"
        );
        let mut blob = [0u8; 4096];
        let mut occupied = [false; 4096];
        let mut len = 0;
        for (arg, p) in args.iter().zip(&self.info.params) {
            let data = match arg {
                Arg::Buffer(b) => {
                    ensure!(
                        std::ptr::eq(b.device, self.module.device),
                        "foreign device buffer"
                    );
                    b.ptr.get().to_ne_bytes().to_vec()
                }
                Arg::U32(n) => n.to_ne_bytes().to_vec(),
                Arg::U64(n) => n.to_ne_bytes().to_vec(),
                Arg::F32(n) => n.to_ne_bytes().to_vec(),
            };
            ensure!(
                p.offset >= 0 && p.size == data.len() as i32,
                "kernel parameter ABI"
            );
            let at = p.offset as usize;
            let end = at + data.len();
            ensure!(
                end <= blob.len() && !occupied[at..end].iter().any(|&v| v),
                "parameter overlap/size"
            );
            blob[at..end].copy_from_slice(&data);
            occupied[at..end].fill(true);
            len = len.max(end);
        }
        Ok(blob[..len].to_vec())
    }
    unsafe fn enqueue_blob(&self, blob: &[u8], grid: [u32; 3], block: [u32; 3]) -> Result<()> {
        // Driver copies parameter bytes during this call; it owns them thereafter.
        unsafe {
            check(tinynv_launch(
                self.module.device.stream,
                self.handle,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                0,
                blob.as_ptr().cast(),
                blob.len(),
            ))
        }
    }
}

pub struct PreparedCall<'a, 'm, 'd> {
    kernel: &'a Kernel<'m, 'd>,
    _args: Vec<Arg<'a, 'd>>,
    blob: Vec<u8>,
    grid: [u32; 3],
    block: [u32; 3],
    /// R65 side branches (r65nv archive): (arrive latch, wait latch, early) applied when this call is
    /// recorded into a graph; None = ordinary serialized edge.
    edge: Option<(u32, u32, bool)>,
}
impl<'a, 'm, 'd> PreparedCall<'a, 'm, 'd> {
    /// R53 calibration: kernel name and launch geometry of this call.
    pub fn meta(&self) -> (&str, [u32; 3], [u32; 3]) {
        (self.kernel.name(), self.grid, self.block)
    }
    /// R65 side branches (r65nv archive): when recorded, this call arrives at latch `arrive` on
    /// completion, waits for latch `wait` (arrived at by an earlier call) and, with `early`, may start
    /// once its predecessor's last CTA has launched. 0 = none. Ignored outside graph capture.
    pub fn with_edge(mut self, arrive: u32, wait: u32, early: bool) -> Self {
        self.edge = Some((arrive, wait, early));
        self
    }
    pub fn set_edge(&mut self, arrive: u32, wait: u32, early: bool) {
        self.edge = Some((arrive, wait, early));
    }
    /// R65: the same call (arguments, launch geometry, edge) on `k`, a kernel with an identical parameter layout
    /// (a variant compiled from the same body, e.g. with a grid dependency wait at entry).
    pub fn with_kernel(self, k: &'a Kernel<'m, 'd>) -> Result<Self> {
        let (a, b) = (&self.kernel.info, &k.info);
        let n = a.num_params.max(0) as usize;
        ensure!(
            a.num_params == b.num_params
                && n <= 64
                && a.params[..n].iter().zip(&b.params[..n]).all(|(x, y)| x.offset == y.offset && x.size == y.size)
                && std::ptr::eq(k.module.device, self.kernel.module.device),
            "R65 kernel swap {} -> {}: parameter layout differs",
            self.kernel.name,
            k.name
        );
        Ok(PreparedCall { kernel: k, ..self })
    }
    pub fn edge(&self) -> Option<(u32, u32, bool)> {
        self.edge
    }
    /// R53 calibration: an identical second launch of this call (same kernel,
    /// arguments and geometry). Only meaningful for kernels that are idempotent.
    pub fn dup(&self) -> PreparedCall<'a, 'm, 'd> {
        PreparedCall {
            kernel: self.kernel,
            _args: self._args.clone(),
            blob: self.blob.clone(),
            grid: self.grid,
            block: self.block,
            edge: None,
        }
    }
    /// R53 calibration: the same geometry launched as a parameterless null kernel.
    /// Calls whose kernel is an R39 early-launch successor (`r39pdl_` prefix) get
    /// `pdl` (which must carry that prefix and wait), all others `plain`, so the
    /// driver records the same dependency edges. No memory is read or written.
    pub fn null_twin<'b, 'n>(
        &self,
        plain: &'b Kernel<'n, 'd>,
        pdl: &'b Kernel<'n, 'd>,
    ) -> Result<PreparedCall<'b, 'n, 'd>> {
        let k = if self.kernel.name.starts_with("r39pdl_") { pdl } else { plain };
        ensure!(
            k.info.num_params == 0 && std::ptr::eq(k.module.device, self.kernel.module.device),
            "null twin kernel"
        );
        Ok(PreparedCall {
            kernel: k,
            _args: vec![],
            blob: vec![],
            grid: self.grid,
            block: self.block,
            edge: None,
        })
    }
    pub fn enqueue(&self) -> Result<()> {
        // Unsafe kernel contract was accepted at construction; buffers/module
        // remain borrowed and immutable launch geometry cannot change.
        unsafe { self.kernel.enqueue_blob(&self.blob, self.grid, self.block) }
    }
}
pub struct Graph<'a, 'm, 'd> {
    device: &'d Device,
    handle: Handle,
    _calls: Vec<PreparedCall<'a, 'm, 'd>>,
}
impl Device {
    pub fn capture<'a, 'm, 'd>(
        &'d self,
        calls: Vec<PreparedCall<'a, 'm, 'd>>,
    ) -> Result<Graph<'a, 'm, 'd>> {
        self.capture_bounded(calls, 128)
    }
    /// Isolated larger-graph experiment. The native driver owns a checked 16MiB
    /// recording region and up to128 chains, rather than a128-call total limit.
    /// Keep ordinary component captures unchanged; require the qualified SYS
    /// ordering control and at most2048 calls for this diagnostic path.
    pub fn capture_model_diagnostic<'a, 'm, 'd>(
        &'d self,
        calls: Vec<PreparedCall<'a, 'm, 'd>>,
    ) -> Result<Graph<'a, 'm, 'd>> {
        ensure!(
            std::env::var("TINYNV_QMD_MEMBAR").as_deref() == Ok("sys"),
            "large diagnostic graph requires qualified SYS ordering"
        );
        self.capture_bounded(calls, 2048)
    }
    fn capture_bounded<'a, 'm, 'd>(
        &'d self,
        calls: Vec<PreparedCall<'a, 'm, 'd>>,
        limit: usize,
    ) -> Result<Graph<'a, 'm, 'd>> {
        ensure!(
            std::env::var("TINYNV_GRAPH_RESIDENT").as_deref() == Ok("1"),
            "resident graph setting required"
        );
        ensure!(
            !calls.is_empty() && calls.len() <= limit,
            "bounded CUDA graph"
        );
        ensure!(
            calls
                .iter()
                .all(|c| std::ptr::eq(c.kernel.module.device, self)),
            "graph foreign device"
        );
        self.sync()?;
        unsafe {
            check(tinynv_graph_begin(self.stream))?;
        }
        // Encoding/ABI validation already completed. Always end a begun capture
        // even if the driver refuses a command; never leave capture mode active.
        let queued = calls.iter().try_for_each(|c| {
            if let Some((arrive, wait, early)) = c.edge {
                #[cfg(feature = "r65-driver")]
                unsafe {
                    check(tinynv_r65_edge(self.stream, arrive, wait, early as std::ffi::c_int))?;
                }
                #[cfg(not(feature = "r65-driver"))]
                anyhow::bail!("R65 edge ({arrive}, {wait}, {early}) needs the r65nv archive");
            }
            c.enqueue()
        });
        let mut handle = null_mut();
        let ended = unsafe { check(tinynv_graph_end(self.stream, &mut handle)) };
        if queued.is_err() || ended.is_err() {
            self.sync()?;
            if !handle.is_null() {
                unsafe {
                    tinynv_graph_free(self.stream, handle);
                }
            }
            queued?;
            ended?;
        }
        ensure!(!handle.is_null(), "null CUDA graph");
        Ok(Graph {
            device: self,
            handle,
            _calls: calls,
        })
    }
}
impl Graph<'_, '_, '_> {
    /// R54 (r53nv archive): rewrite one 32-bit argument of recorded launch `launch`
    /// (index in capture order). Applied by the next replay before any of its
    /// launches and kept for later replays. Only U32 arguments may be patched.
    #[cfg(feature = "r53-driver")]
    pub fn patch_u32(&self, launch: usize, arg: usize, value: u32) -> Result<()> {
        ensure!(!self.handle.is_null(), "closed CUDA graph");
        let call = self._calls.get(launch).ok_or_else(|| anyhow::anyhow!("graph patch launch {launch}"))?;
        ensure!(matches!(call._args.get(arg), Some(Arg::U32(_))), "graph patch argument {arg} is not U32");
        let p = &call.kernel.info.params[arg];
        ensure!(p.offset >= 0 && p.size == 4, "graph patch parameter ABI");
        unsafe {
            check(tinynv_r54_graph_patch(self.device.stream, self.handle, launch as u32, p.offset as u32,
                (&value as *const u32).cast(), 4))
        }
    }
    pub fn launch(&self) -> Result<()> {
        ensure!(!self.handle.is_null(), "closed CUDA graph");
        unsafe { check(tinynv_graph_launch(self.device.stream, self.handle)) }
    }
    fn release(&mut self) -> Result<()> {
        if !self.handle.is_null() {
            self.device.sync()?;
            unsafe {
                tinynv_graph_free(self.device.stream, self.handle);
            }
            self.handle = null_mut();
        }
        Ok(())
    }
}
impl<'a, 'm, 'd> Graph<'a, 'm, 'd> {
    /// R56: re-target this captured graph to `new`, a fresh recording of the same plan (same
    /// kernels, geometry and buffers in capture order): every scalar argument that differs is
    /// patched at its parameter offset (applied by the next replay before any launch, kept for
    /// later replays) and `new` becomes the recorded state. Returns None, patching nothing, if
    /// the structure differs (the caller re-captures).
    #[cfg(feature = "r53-driver")]
    pub fn patch_diff(&mut self, new: Vec<PreparedCall<'a, 'm, 'd>>) -> Result<Option<usize>> {
        if new.len() != self._calls.len() {
            return Ok(None);
        }
        self.patch_range(0, new)
    }
    /// R56: patch_diff for the recorded launches [start, start + new.len()).
    #[cfg(feature = "r53-driver")]
    pub fn patch_range(&mut self, start: usize, new: Vec<PreparedCall<'a, 'm, 'd>>) -> Result<Option<usize>> {
        ensure!(!self.handle.is_null(), "closed CUDA graph");
        if start + new.len() > self._calls.len() {
            return Ok(None);
        }
        for (a, b) in self._calls[start..].iter().zip(&new) {
            if !std::ptr::eq(a.kernel, b.kernel) || a.grid != b.grid || a.block != b.block || a._args.len() != b._args.len() {
                return Ok(None);
            }
            for (x, y) in a._args.iter().zip(&b._args) {
                let same_kind = match (x, y) {
                    (Arg::Buffer(p), Arg::Buffer(q)) => p.ptr.get() == q.ptr.get(),
                    (Arg::U32(_), Arg::U32(_)) | (Arg::U64(_), Arg::U64(_)) | (Arg::F32(_), Arg::F32(_)) => true,
                    _ => false,
                };
                if !same_kind {
                    return Ok(None);
                }
            }
        }
        let mut n = 0;
        for (o, b) in new.iter().enumerate() {
            let i = start + o;
            for (j, y) in b._args.iter().enumerate() {
                let bytes: Option<Vec<u8>> = match (&self._calls[i]._args[j], y) {
                    (Arg::U32(p), Arg::U32(q)) if p != q => Some(q.to_le_bytes().to_vec()),
                    (Arg::U64(p), Arg::U64(q)) if p != q => Some(q.to_le_bytes().to_vec()),
                    (Arg::F32(p), Arg::F32(q)) if p.to_bits() != q.to_bits() => Some(q.to_le_bytes().to_vec()),
                    _ => None,
                };
                if let Some(bytes) = bytes {
                    let p = &b.kernel.info.params[j];
                    ensure!(p.offset >= 0 && p.size as usize == bytes.len(), "graph diff parameter ABI");
                    unsafe {
                        check(tinynv_r54_graph_patch(self.device.stream, self.handle, i as u32, p.offset as u32,
                            bytes.as_ptr().cast(), bytes.len() as u32))?;
                    }
                    n += 1;
                }
            }
        }
        let end = start + new.len();
        self._calls.splice(start..end, new);
        Ok(Some(n))
    }
}
impl Drop for Graph<'_, '_, '_> {
    fn drop(&mut self) {
        if let Err(e) = self.release() {
            eprintln!("CUDA graph cleanup: {e:#}");
        }
    }
}
