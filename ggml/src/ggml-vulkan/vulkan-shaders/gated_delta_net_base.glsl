#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : enable
#if USE_SUBGROUP_CLUSTERED
#extension GL_KHR_shader_subgroup_clustered : enable
#endif
#if USE_SUBGROUP_ADD
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

// Caller guarantees valid spec constants: S_V % COLS_PER_WG == 0 and S_V % LANES_PER_COLUMN == 0,
// so no bounds checking is needed.
layout(constant_id = 0) const uint S_V = 128;
layout(constant_id = 1) const uint KDA = 0;
layout(constant_id = 2) const uint SUBGROUP_SIZE = 32;
layout(constant_id = 3) const uint LANES_PER_COLUMN = 32;

const uint COLS_PER_WG = SUBGROUP_SIZE / LANES_PER_COLUMN;
const uint ROWS_PER_LANE = S_V / LANES_PER_COLUMN;

layout(binding = 0) readonly  buffer QBuf     { FLOAT_TYPE data_q[];     };
layout(binding = 1) readonly  buffer KBuf     { FLOAT_TYPE data_k[];     };
layout(binding = 2) readonly  buffer VBuf     { FLOAT_TYPE data_v[];     };
layout(binding = 3) readonly  buffer GBuf     { FLOAT_TYPE data_g[];     };
layout(binding = 4) readonly  buffer BetaBuf  { FLOAT_TYPE data_beta[];  };
layout(binding = 5) readonly  buffer StateBuf { FLOAT_TYPE data_state[]; };

#if !USE_SUBGROUP_ADD && !USE_SUBGROUP_CLUSTERED
shared FLOAT_TYPE temp[SUBGROUP_SIZE];

// This does a reduction across groups of LANES_PER_COLUMN
FLOAT_TYPE reduce_add_shmem(FLOAT_TYPE partial) {
    const uint lane = gl_SubgroupInvocationID;
    temp[lane] = partial;
    barrier();
    [[unroll]] for (uint s = LANES_PER_COLUMN / 2u; s > 0; s >>= 1u) {
        FLOAT_TYPE other = temp[lane ^ s];
        barrier();
        temp[lane] += other;
        barrier();
    }
    const FLOAT_TYPE result = temp[lane];
    barrier();
    return result;
}
#endif

// clusterSize for subgroupClusteredAdd must be a compile-time constant; branch on spec constant
FLOAT_TYPE reduce_partial(FLOAT_TYPE partial) {
    switch (LANES_PER_COLUMN) {
        case 1u:
            return partial;
#if USE_SUBGROUP_CLUSTERED
        // Workaround for GLSL requiring a literal constant for the cluster size.
        // The branches should all fold away.
        case 2u:
            return subgroupClusteredAdd(partial, 2u);
        case 4u:
            return subgroupClusteredAdd(partial, 4u);
        case 8u:
            return subgroupClusteredAdd(partial, 8u);
        case 16u:
            return subgroupClusteredAdd(partial, 16u);
        case 32u:
            return subgroupClusteredAdd(partial, 32u);
        case 64u:
            return subgroupClusteredAdd(partial, 64u);
#endif
        default:
#if USE_SUBGROUP_ADD
            return subgroupAdd(partial);
#else
            return reduce_add_shmem(partial);
#endif
    }
}
