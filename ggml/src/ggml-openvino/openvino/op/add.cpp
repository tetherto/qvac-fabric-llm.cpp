#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/util/precision_sensitive_attribute.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_add(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    if (context.get_op_case() == 1) {
        // MoE expert-plane sum (see is_moe_expert_sum_add): input 1 is a VIEW plane of the
        // shared base tensor `experts` = [n_embd, n_expert_used, n_tokens, 1] (ggml order) ->
        // [1, n_tokens, n_expert_used, n_embd] (OV order). The whole ADD chain is equivalent to
        // reducing the expert axis (OV axis 2) of that base, so bypass the chain and the
        // per-plane Slices entirely.
        size_t view_size = context.get_view_input_size(1);
        auto base_name = context.get_view_input_src_name(1, view_size - 1);
        auto base = context.get_input(base_name);

        auto reduced = std::make_shared<ov::op::v1::ReduceSum>(
            base, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}), false);
        auto res =
            std::make_shared<ov::op::v0::Unsqueeze>(reduced, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}));
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    auto input_0 = process_view_input_new(context, 0);
    auto input_1 = process_view_input_new(context, 1);

    // ggml allows ADD with mismatched src types (e.g. an f16 residual plus an
    // f32 addend): it adds in f32 and rounds once into the dst type. OV Add
    // requires matching input types, so upcast to f32, add, and convert to the
    // dst type. The precision-sensitive marks keep the GPU plugin from
    // compressing the f32 math back to f16, which would round the addend a
    // second time.
    const bool mixed_types = input_0.get_element_type() != input_1.get_element_type();
    if (mixed_types) {
        input_0 = std::make_shared<ov::op::v0::Convert>(input_0, ov::element::f32);
        input_1 = std::make_shared<ov::op::v0::Convert>(input_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(input_0, input_1);

    if (mixed_types) {
        ov::mark_as_precision_sensitive(res.get_node_shared_ptr()->input(0));
        ov::mark_as_precision_sensitive(res.get_node_shared_ptr()->input(1));

        const auto output_type = context.get_output_type();
        if (res.get_element_type() != output_type) {
            auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
            ov::mark_as_precision_sensitive(output_convert->input(0));
            res = output_convert;
        }
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
