//! Machine identity read from sysctl.

const std = @import("std");

var chip_brand_buf: [128]u8 = undefined;
var chip_brand_len: usize = 0;
/// 0 = unread, 1 = one thread is reading it, 2 = published.
var chip_brand_state = std.atomic.Value(u8).init(0);

/// Cached `machdep.cpu.brand_string` ("Apple M3 Ultra"); empty on failure, and
/// callers then fall to their default row. THE accessor for every per-silicon
/// table (MTP depth cap, DFlash block cap): the GPU arch string cannot tell
/// Ultra from Max, and the sysctl runs exactly once.
pub fn chipBrand() []const u8 {
    if (chip_brand_state.load(.acquire) != 2) {
        if (chip_brand_state.cmpxchgStrong(0, 1, .acquire, .monotonic) == null) {
            chip_brand_len = chipBrandString(&chip_brand_buf).len;
            chip_brand_state.store(2, .release);
        } else {
            while (chip_brand_state.load(.acquire) != 2) std.atomic.spinLoopHint();
        }
    }
    return chip_brand_buf[0..chip_brand_len];
}

fn chipBrandString(buf: []u8) []const u8 {
    var len: usize = buf.len;
    if (std.c.sysctlbyname("machdep.cpu.brand_string", buf.ptr, &len, null, 0) != 0) return "";
    if (len > 0 and buf[len - 1] == 0) len -= 1;
    return buf[0..len];
}
