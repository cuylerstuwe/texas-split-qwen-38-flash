//! R56: persistent reader threads for the PLE n-gram tables' random row reads (32 GB drafter
//! table, 102 GB target table: cold rows cost SSD latency). Two lanes: `read_groups` (on the
//! critical path; caller waits) and `prefetch` (fire-and-forget page-cache warm-up). Each lane
//! has its own threads, so prefetches never queue in front of a waiting gather.
use std::{
    fs::File,
    os::unix::fs::FileExt,
    sync::{Arc, Mutex, OnceLock, mpsc},
};

#[derive(Clone)]
pub struct Read {
    pub file: Arc<File>,
    pub at: u64,
    pub len: usize,
}
type Reply = mpsc::Sender<(usize, std::io::Result<Vec<Vec<u8>>>)>;
type Job = (Vec<Read>, Option<Reply>, usize);

struct Lane {
    tx: Mutex<mpsc::Sender<Job>>,
}
fn lane(threads: usize) -> Lane {
    let (tx, rx) = mpsc::channel::<Job>();
    let rx = Arc::new(Mutex::new(rx));
    for _ in 0..threads {
        let rx = rx.clone();
        std::thread::spawn(move || {
            loop {
                let job = rx.lock().unwrap().recv();
                let Ok((reads, back, tag)) = job else { break };
                let out: std::io::Result<Vec<Vec<u8>>> = reads
                    .iter()
                    .map(|r| {
                        let mut b = vec![0u8; r.len];
                        r.file.read_exact_at(&mut b, r.at).map(|_| b)
                    })
                    .collect();
                if let Some(back) = back {
                    let _ = back.send((tag, out));
                }
            }
        });
    }
    Lane { tx: Mutex::new(tx) }
}
/// Critical-lane threads: QWEN_IO_THREADS (read once; default 16).
fn critical() -> &'static Lane {
    static L: OnceLock<Lane> = OnceLock::new();
    L.get_or_init(|| lane(std::env::var("QWEN_IO_THREADS").ok().and_then(|v| v.parse().ok()).filter(|&n: &usize| (1..=256).contains(&n)).unwrap_or(16)))
}
fn background() -> &'static Lane {
    static L: OnceLock<Lane> = OnceLock::new();
    L.get_or_init(|| lane(8))
}
/// Read every group concurrently (one job per group); results in group order.
pub fn read_groups(groups: Vec<Vec<Read>>) -> std::io::Result<Vec<Vec<Vec<u8>>>> {
    let n = groups.len();
    let (back, rx) = mpsc::channel();
    {
        let tx = critical().tx.lock().unwrap();
        for (i, g) in groups.into_iter().enumerate() {
            tx.send((g, Some(back.clone()), i)).expect("io pool");
        }
    }
    drop(back);
    let mut out: Vec<Option<Vec<Vec<u8>>>> = (0..n).map(|_| None).collect();
    for _ in 0..n {
        let (i, r) = rx.recv().expect("io pool reply");
        out[i] = Some(r?);
    }
    Ok(out.into_iter().map(|o| o.expect("io pool group")).collect())
}
/// Warm the page cache for these reads on the background lane; returns at once.
pub fn prefetch(reads: Vec<Read>) {
    let tx = background().tx.lock().unwrap();
    for r in reads {
        let _ = tx.send((vec![r], None, 0));
    }
}

unsafe extern "C" {
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
}
const F_RDADVISE: i32 = 44;
const F_RDAHEAD: i32 = 45;
#[repr(C)]
struct Radvisory {
    ra_offset: i64,
    ra_count: i32,
}
/// Start reading this range into the page cache and return at once (F_RDADVISE: the kernel's
/// advisory read, no copy to user space). Errors are ignored: it is only a hint.
pub fn advise(file: &File, at: u64, len: usize) {
    use std::os::fd::AsRawFd;
    let ra = Radvisory { ra_offset: at as i64, ra_count: len as i32 };
    unsafe {
        fcntl(file.as_raw_fd(), F_RDADVISE, &ra as *const Radvisory);
    }
}
/// Read-ahead for this open file (F_RDAHEAD): random row reads gain nothing from it.
pub fn set_readahead(file: &File, on: bool) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    if unsafe { fcntl(file.as_raw_fd(), F_RDAHEAD, on as i32) } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// R56: gather latency counters (host wall time per gather), read and reset by `take`.
pub struct Stats {
    n: std::sync::atomic::AtomicU64,
    us: std::sync::atomic::AtomicU64,
    slow: std::sync::atomic::AtomicU64,
    max: std::sync::atomic::AtomicU64,
}
impl Stats {
    pub const fn new() -> Self {
        use std::sync::atomic::AtomicU64;
        Self { n: AtomicU64::new(0), us: AtomicU64::new(0), slow: AtomicU64::new(0), max: AtomicU64::new(0) }
    }
    pub fn record(&self, us: f64) {
        use std::sync::atomic::Ordering::Relaxed;
        let u = us as u64;
        self.n.fetch_add(1, Relaxed);
        self.us.fetch_add(u, Relaxed);
        if u > 500 {
            self.slow.fetch_add(1, Relaxed);
        }
        self.max.fetch_max(u, Relaxed);
    }
    /// {gathers, total_ms, mean_us, over_500us, max_us}; resets.
    pub fn take(&self) -> serde_json::Value {
        use std::sync::atomic::Ordering::Relaxed;
        let (n, us, slow, max) = (self.n.swap(0, Relaxed), self.us.swap(0, Relaxed), self.slow.swap(0, Relaxed), self.max.swap(0, Relaxed));
        serde_json::json!({"gathers":n,"total_ms":us as f64/1e3,"mean_us":if n > 0 {us as f64/n as f64} else {0.},"over_500us":slow,"max_us":max})
    }
}
/// The drafter's (ple::Table) and the Metal target's (metal_ple_target::Table) gathers.
pub static DRAFTER_GATHER: Stats = Stats::new();
pub static TARGET_GATHER: Stats = Stats::new();

unsafe extern "C" {
    fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
    fn msync(addr: *mut std::ffi::c_void, len: usize, flags: i32) -> i32;
}
/// R56: drop this file's clean pages from the page cache (a read-only shared mapping, then
/// msync(MS_INVALIDATE), which on macOS evicts them; F_NOCACHE reads still left ~87% cached).
/// For weights already copied into device memory: their cached pages only crowd out the n-gram
/// tables and the Metal target's mapped experts. Returns the file's size.
pub fn evict_cached(path: &std::path::Path) -> std::io::Result<u64> {
    use std::os::fd::AsRawFd;
    let f = File::open(path)?;
    let len = f.metadata()?.len() as usize;
    if len == 0 {
        return Ok(0);
    }
    const PROT_READ: i32 = 1;
    const MAP_SHARED: i32 = 1;
    const MS_INVALIDATE: i32 = 2;
    unsafe {
        let p = mmap(std::ptr::null_mut(), len, PROT_READ, MAP_SHARED, f.as_raw_fd(), 0);
        if p as isize == -1 {
            return Err(std::io::Error::last_os_error());
        }
        let rc = msync(p, len, MS_INVALIDATE);
        let err = std::io::Error::last_os_error();
        munmap(p, len);
        if rc != 0 {
            return Err(err);
        }
    }
    Ok(len as u64)
}

unsafe extern "C" {
    fn getpagesize() -> i32;
    fn mincore(addr: *const std::ffi::c_void, len: usize, vec: *mut u8) -> i32;
}
const F_NOCACHE: i32 = 48;
/// R56: bytes [at, at+len) of `path` without leaving them in the page cache. macOS honours
/// F_NOCACHE only for direct reads whose file offset, length and buffer are page aligned
/// (Apple DTS; XNU vfs_cluster.c cluster_read_direct: a misaligned request takes the cached
/// copy path), so whole aligned pages are read into a page-aligned staging buffer and the
/// requested bytes copied out (the approach of kapitsa PR 945). Pages that were already resident
/// are served from the cache and stay there (use `evict_cached` for those).
pub fn read_uncached(path: &std::path::Path, at: u64, len: usize) -> std::io::Result<Vec<u8>> {
    use std::os::fd::AsRawFd;
    const CHUNK: usize = 8 << 20;
    let page = unsafe { getpagesize() } as u64;
    let f = File::open(path)?;
    if unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1i32) } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    let size = f.metadata()?.len();
    let end = at.checked_add(len as u64).filter(|&e| e <= size).ok_or_else(|| std::io::Error::other("read past end of file"))?;
    let layout = std::alloc::Layout::from_size_align(CHUNK, page as usize).map_err(std::io::Error::other)?;
    let staging = unsafe { std::alloc::alloc(layout) };
    if staging.is_null() {
        return Err(std::io::Error::other("staging allocation"));
    }
    struct Free(*mut u8, std::alloc::Layout);
    impl Drop for Free {
        fn drop(&mut self) {
            unsafe { std::alloc::dealloc(self.0, self.1) }
        }
    }
    let _free = Free(staging, layout);
    let buf = unsafe { std::slice::from_raw_parts_mut(staging, CHUNK) };
    let mut out = Vec::with_capacity(len);
    let mut pos = at - at % page;
    while pos < end {
        // whole pages; the file's last partial page is the only short read
        let want = (CHUNK as u64).min(end.div_ceil(page) * page - pos) as usize;
        let mut got = 0;
        while got < want {
            let n = f.read_at(&mut buf[got..want], pos + got as u64)?;
            if n == 0 {
                break;
            }
            got += n;
        }
        let lo = at.max(pos);
        let hi = end.min(pos + got as u64);
        if hi <= lo {
            return Err(std::io::Error::other("short uncached read"));
        }
        out.extend_from_slice(&buf[(lo - pos) as usize..(hi - pos) as usize]);
        pos += want as u64;
    }
    if out.len() != len {
        return Err(std::io::Error::other("uncached read length"));
    }
    Ok(out)
}
/// R56 diagnostics: fraction of `path`'s pages resident in the page cache (mincore).
pub fn resident_fraction(path: &std::path::Path) -> std::io::Result<f64> {
    use std::os::fd::AsRawFd;
    let f = File::open(path)?;
    let len = f.metadata()?.len() as usize;
    if len == 0 {
        return Ok(0.);
    }
    let page = unsafe { getpagesize() } as usize;
    let pages = len.div_ceil(page);
    let mut vec = vec![0u8; pages];
    unsafe {
        let p = mmap(std::ptr::null_mut(), len, 1, 1, f.as_raw_fd(), 0);
        if p as isize == -1 {
            return Err(std::io::Error::last_os_error());
        }
        let rc = mincore(p, len, vec.as_mut_ptr());
        let err = std::io::Error::last_os_error();
        munmap(p, len);
        if rc != 0 {
            return Err(err);
        }
    }
    Ok(vec.iter().filter(|&&v| v & 1 != 0).count() as f64 / pages as f64)
}
