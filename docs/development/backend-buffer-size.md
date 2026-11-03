# Allocation-free buffer sizing

`ggml_backend_buft_get_buffer_alloc_size(buft, payload_bytes)` predicts the
physical size of a single buffer allocated for that payload request. The
optional registry procedure of the same name implements the conversion;
without it the conversion is the identity. It does not change the backend
interface layout or the tensor-level `get_alloc_size` contract.

The callback must be allocation-free, must not mutate backend state, and must
return at least the requested payload size. `SIZE_MAX` denotes an
unrepresentable request. A zero request always reports zero: the public
allocation wrapper returns an empty metadata buffer without invoking the
backend allocator. This value describes backend-reported buffer bytes, not
unreported driver heaps, process RSS or all physical memory in use.

Context sizing (`ggml_backend_alloc_ctx_tensors_from_buft_size`) and graph
reservation (`ggml_gallocr_reserve_n_size`) apply this conversion to each of
the actual allocator's payload chunks, then add the results with saturation.
The number of chunks comes from normal packing and liveness analysis; it is
not inferred from total bytes or charged once per component. Repeated graph
buffer types share an allocator and are charged only at their first index,
matching the allocated-buffer query. Tensor payloads are never allocated by
the sizing functions.

Buffers keep reporting their payload size through `ggml_backend_buffer_get_size`.
Graph growth and shared-buffer detachment therefore compare payloads with
payloads; the guard lives only in the physical backing allocation.

The Hexagon implementation shares one checked helper between its shared-buffer
constructor and the query: the payload rounded up to a page, one guard page,
and the DMA64 alignment for unpinned buffers. Both the device and the host
buffer type allocate unpinned shared buffers, so one procedure serves both.

`test-backend-buffer-size` uses a host-portable synthetic backend with a
small payload cap and a non-page guard. It checks fragmented context and graph
allocations against actual backing buffers, repeated buffer types, views,
empty and already-allocated contexts, zero-size dummy buffers, guard overflow,
saturating totals, graph growth and shared scheduler growth with
source-payload preservation and exact buffer lifetimes after detachment.
The existing CPU memory regression (`test-backend-memory`) continues to check
identity-backend byte parity.
