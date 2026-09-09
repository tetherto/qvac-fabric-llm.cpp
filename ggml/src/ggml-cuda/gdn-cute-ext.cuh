/*
 * Copyright (c) 2025 by FlashInfer team.
 * Source: https://github.com/flashinfer-ai/flashinfer/blob/992848ad3d0e21650f2d6cd2e03becae187cf743/include/flashinfer/flat/cute_ext.hpp
 * Modified for llama.cpp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cute/tensor.hpp"

namespace flat {

using namespace cute;

template <int... Is, typename Tensor>
__forceinline__ __host__ __device__ constexpr auto select_tensor(Tensor&& t) {
  if constexpr (is_composed_layout<decltype(t.layout())>::value) {
    return make_tensor(
        std::forward<Tensor>(t).data(),
        make_composed_layout(std::forward<Tensor>(t).layout().layout_a(),
                             std::forward<Tensor>(t).layout().offset(),
                             select<Is...>(std::forward<Tensor>(t).layout().layout_b())));
  } else {
    return make_tensor(std::forward<Tensor>(t).data(), select<Is...>(t.layout()));
  }
}

}  // namespace flat
