// Upper bound on the launch, and the size of the scratch it reserves. The host clamps
// the work-group to the kernel's own CL_KERNEL_WORK_GROUP_SIZE, which can be smaller, so
// the reduction below is driven by get_local_size rather than by this.
#define ARGMAX_WG 256

// One workgroup reduces one row. Ties resolve to the largest index, matching
// ggml_vec_argmax_f32, which reassigns whenever the running max equals the element.
kernel void kernel_argmax_f32(
        global void * src0,
        ulong offset0,
        global void * dst,
        ulong offsetd,
        int ne00,
        ulong nb01
) {
    local float shmax[ARGMAX_WG];
    local int   shidx[ARGMAX_WG];

    const int row  = get_group_id(0);
    const int lid  = get_local_id(0);
    const int nloc = (int)get_local_size(0);

    global const float * src = (global const float *)((global char *)src0 + offset0 + (ulong)row*nb01);
    global int * out = (global int *)((global char *)dst + offsetd);

    float best = -INFINITY;
    int   besti = 0;
    for (int i = lid; i < ne00; i += nloc) {
        const float v = src[i];
        if (v >= best) {
            best  = v;
            besti = i;
        }
    }

    shmax[lid] = best;
    shidx[lid] = besti;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Fold from the next power of two at or above nloc, guarding the upper half, so a
    // work-group that is not a power of two still folds every slot exactly once and no
    // slot past the launch is ever read.
    int s = 1;
    while (s < nloc) {
        s <<= 1;
    }
    for (s >>= 1; s > 0; s >>= 1) {
        if (lid < s && lid + s < nloc) {
            const float o  = shmax[lid + s];
            const int   oi = shidx[lid + s];
            if (o > shmax[lid] || (o == shmax[lid] && oi > shidx[lid])) {
                shmax[lid] = o;
                shidx[lid] = oi;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[row] = shidx[0];
    }
}
