# Texas native foreground/background overlay

This local overlay ports Texas's live-process priority contract onto Sushi 1.3.0.
The OpenAI-facing supervisor owns authenticated, independent controller leases
and direct foreground tickets. A private atomic IPC page connects those holds to
the sole inference owner. No MLX operation runs in the supervisor or a control
thread. Without the private IPC environment, ordinary Sushi behavior remains.

The owner suspends one background stack at a completed prefill chunk or decode
boundary, drains the MLX stream, releases per-request MTP/SSM ownership from shared
modules, and pumps foreground work on that same owner. Texas uses one foreground
request at a time. A queued direct foreground request or media encode also holds
the local gate. Background work never joins a foreground batch.

All logical request state, grammar, RNG, stream state and unfinished prefill
locals remain live. Large materialized request allocations, including ancestors of pending lazy
request graphs, may park losslessly on SSD. Symbolic leaves remain untouched.
Protection traverses model/private/published-cache roots and their lazy
dependencies without evaluating symbolic leaves. Published roots are retained
while nested foreground work runs, and unfinished background write-through
publication is deferred until completion/cancellation.

Parking preserves MLX Data identity, offsets, strides and aliases. Every write
and fsync completes before any buffer is released. Each allocation carries a
SHA-256 checksum verified before restored data is published. Write failure keeps
the original RAM. Truncation, corruption or unrecoverable allocation failure
terminates before incomplete tensors execute. Checkpoints are exclusive 0600,
cache-bypassed files beneath a locked 0700 directory, with a 24 GiB request cap
and 32 GiB additional disk reserve.

Restoration waits for 12 GiB reclaimable supply plus parked bytes. New holds
during restoration require re-parking before a drained acknowledgement. Owner
acknowledgements carry the current IPC sequence; an idle observation by a
controller is not sufficient. Native idle waits retain foreground wakeup,
residency keepalive and sleep-inhibition policy.

MLX's hidden per-dylib random TLS is not the helper's inline `default_()` state.
The checkpoint helper resolves the pinned library's TLV descriptors by symbol
name and saves/restores its actual live KeySequence. This is a private MLX ABI
dependency, like Data buffer mutation; a changed or stripped library requires
requalification. Missing symbols fail closed. Native fixtures exercise the real
library RNG, alias/stride/weight protection, partial writes, truncated reads and
checksum corruption.

Deployment pins, limits and watchdogs belong to the isolated Texas host. The
canonical model and original Sushi MLX dylibs/metallib are reused in place.
Crash/reboot recovery of an in-flight request and background media are outside
this live-process contract.
