// Packed 4x8-bit integer dot products. A glslc without GL_EXT_integer_dot_product emits the same SPIR-V instructions
// (OpSDot and OpSDotAccSat with the packed 4x8-bit format) through GL_EXT_spirv_intrinsics instead.
#if defined(INTEGER_DOT_SPIRV_INTRINSICS)
#extension GL_EXT_spirv_intrinsics : require

#define SPV_OP_SDOT 4450
#define SPV_OP_SDOT_ACC_SAT 4453
#define SPV_CAPABILITY_DOT_PRODUCT_INPUT_4X8_BIT_PACKED 6018
#define SPV_CAPABILITY_DOT_PRODUCT 6019
#define SPV_PACKED_VECTOR_FORMAT_4X8_BIT 0

spirv_instruction(extensions = ["SPV_KHR_integer_dot_product"],
                  capabilities = [SPV_CAPABILITY_DOT_PRODUCT, SPV_CAPABILITY_DOT_PRODUCT_INPUT_4X8_BIT_PACKED],
                  id = SPV_OP_SDOT)
int spv_sdot_packed(int a, int b, spirv_literal int format);

spirv_instruction(extensions = ["SPV_KHR_integer_dot_product"],
                  capabilities = [SPV_CAPABILITY_DOT_PRODUCT, SPV_CAPABILITY_DOT_PRODUCT_INPUT_4X8_BIT_PACKED],
                  id = SPV_OP_SDOT_ACC_SAT)
int spv_sdot_acc_sat_packed(int a, int b, int acc, spirv_literal int format);

#define dotPacked4x8EXT(a, b) spv_sdot_packed(a, b, SPV_PACKED_VECTOR_FORMAT_4X8_BIT)
#define dotPacked4x8AccSatEXT(a, b, acc) spv_sdot_acc_sat_packed(a, b, acc, SPV_PACKED_VECTOR_FORMAT_4X8_BIT)
#else
#extension GL_EXT_integer_dot_product : require
#endif
