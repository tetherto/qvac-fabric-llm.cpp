# Allocation-free buffer sizing

`ggml_backend_buft_get_buffer_alloc_size(buft, payload_bytes)` reports the
physical size of a single buffer that `alloc_buffer` would create for that
payload request. The optional registry procedure of the same name implements
the conversion; without it the function is the identity. It does not change
the backend interface layout, the tensor-level `get_alloc_size` contract or
any size estimate the core allocator reports.

The procedure must be allocation-free, must not mutate backend state, and must
return at least the requested payload size. `SIZE_MAX` denotes an
unrepresentable request. A zero request always reports zero: the public
allocation wrapper returns an empty metadata buffer without invoking the
backend allocator. The value describes backend-reported buffer bytes, not
unreported driver heaps, process RSS or all physical memory in use.

Buffers keep reporting their payload size through `ggml_backend_buffer_get_size`;
the guard lives only in the physical backing allocation. Callers that price
their own buffers, such as the speech engines' weight repack measurement, ask
this function once per buffer they would allocate.

The Hexagon implementation shares one checked helper between its shared-buffer
constructor and the procedure: the payload rounded up to a page, one guard page,
and the DMA64 alignment for unpinned buffers. Both the device and the host
buffer type allocate unpinned shared buffers, so one procedure serves both.

`test-backend-buffer-size` uses a host-portable synthetic buffer type with a
guard the buffer does not report. It checks that the query matches the
allocator's physical allocation without allocating, that a zero request stays
zero, that guard overflow saturates, and that a backend without the procedure
is the identity.
