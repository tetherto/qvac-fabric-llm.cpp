# Hexagon runtime information

`ggml-hexagon.h` is installed even when the Hexagon backend is disabled. Consumers
look up `ggml_backend_hexagon_get_runtime_info_v1` through the backend registry;
a missing procedure means the capability is unavailable.

Pass an initialized Hexagon backend, the maximum prospective graph node count,
and the exact size of `ggml_backend_hexagon_runtime_info_v1`. Invalid arguments,
overflow, unsupported node counts, or a query made from inside a backend operation
return false without modifying the output.

The query snapshots all initialized registry sessions, including child sessions
of a virtual device. It does not initialize other devices. Session count, retained
queue configuration, and execution mode describe the initialized backend, rather
than a fresh read of environment variables. Fabric submits and executes DSP work,
so `op_stage` is 3.

Host metadata accounting includes session and device objects, retained vector
capacities, hash buckets and entries, graph nodes and their auxiliary arrays,
queue caches, fences, and cloned-buffer wrappers. Hash entries include two pointer
widths of linkage allowance. Prospective graphs reserve four doubled-capacity node
copies, plus doubled-capacity queued batches. Each projected node includes 64
pointer slots, two dummy tensor objects, and 512 name bytes. These are an explicit
conservative allocation policy, not a measurement of process RSS or an allocator's
internal overhead.

Shared memory includes page-rounded queue storage, fences, and the retained DSP
request/response queue sizes. Tensor allocations, model weights, allocator
bookkeeping, opaque vendor heaps and driver allocations are excluded. Callers must
retain a memory margin. VTCM and batch virtual-address limits are separate limits,
not additional host RAM pools. Maximum buffer payload uses the retained dynamic-buffer limit and excludes the
tensor guard. The separate static-weight packing and total-buffer limits are not
additional allocations in this estimate.

A snapshot excludes metadata mutation while it is being collected. Independent
backend operations can run concurrently; callers still must obey existing backend
thread-safety requirements and serialize subsequent work if they need the snapshot
to remain valid. The query does not imply that outstanding DSP execution has ended.

Run `test-hexagon-runtime-info` for host arithmetic and snapshot exclusion checks.
On an HTP device, run `test-hexagon-runtime-info-device` to check the registry ABI,
invalid arguments, projected metadata, and retained environment configuration.
The device test exits 77 when no HTP device is available.
