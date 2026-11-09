//! R56: the calling thread's macOS scheduling class, recorded per run and raised to
//! user-interactive (a per-thread hint inside this process only; no system setting). Two early
//! hybrid runs had 20-40% slower host-side work; the probes turned out to inherit
//! user-interactive already (r56-hybrid-roll-sample-03..05: 74.5/74.7/74.9 tok/s), so that
//! variance came from elsewhere (likely other activity on the Mac at the time).
#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn qos_class_self() -> u32;
    fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
}
pub const USER_INTERACTIVE: u32 = 0x21;
pub fn name(q: u32) -> &'static str {
    match q {
        0x21 => "user_interactive",
        0x19 => "user_initiated",
        0x15 => "default",
        0x11 => "utility",
        0x09 => "background",
        _ => "unspecified",
    }
}
/// Returns (inherited, now) QoS classes of the calling thread.
#[cfg(target_os = "macos")]
pub fn raise_to_interactive() -> anyhow::Result<(u32, u32)> {
    let before = unsafe { qos_class_self() };
    let rc = unsafe { pthread_set_qos_class_self_np(USER_INTERACTIVE, 0) };
    anyhow::ensure!(rc == 0, "pthread_set_qos_class_self_np failed: {rc}");
    Ok((before, unsafe { qos_class_self() }))
}
