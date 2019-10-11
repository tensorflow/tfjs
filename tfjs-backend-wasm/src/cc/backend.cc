/* Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <xnnpack.h>
#include <unordered_map>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace {
// Maps a unique tensor id to info about that tensor. The map owns all of its
// entries.
std::unordered_map<int, TensorInfo> data;

// Maps a tensor id to a vector of disposal functions registered on that tensor
// id.
std::unordered_map<int, std::vector<tfjs::backend::DisposeFunction>>
    disposal_callbacks;
}  // namespace

namespace tfjs {
namespace backend {
TensorInfo get_tensor_info(int tensor_id) { return data.at(tensor_id); }

int xnn_operator_count = 0;

// Registers a disposal callback for a tensor id with a given callback function.
void register_disposal_callback(int tensor_id, DisposeFunction dispose_fn) {
  if (disposal_callbacks.count(tensor_id) == 0) {
    auto callbacks = std::vector<DisposeFunction>{dispose_fn};
    // We move callbacks to avoid a copy.
    disposal_callbacks.insert({tensor_id, std::move(callbacks)});
  } else {
    auto callbacks = disposal_callbacks.at(tensor_id);
    callbacks.push_back(dispose_fn);
  }
}

int num_tensors() { return data.size(); }

}  // namespace backend

namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void init() { xnn_initialize(); }

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void register_tensor(int tensor_id, int *shape_ptr, int shape_length,
                     DType dtype, void *memory_offset) {
  auto shape = std::vector<int>(shape_ptr, shape_ptr + shape_length);
  auto size = util::size_from_shape(shape);

  TensorInfo info = {{}, dtype, std::move(shape), size};
  switch (dtype) {
    case DType::float32:
      info.buf.f32 = static_cast<float *>(memory_offset);
      break;
    case DType::int32:
      info.buf.i32 = static_cast<int *>(memory_offset);
      break;
    case DType::boolean:
      info.buf.b = static_cast<bool *>(memory_offset);
      break;
    default:
      util::warn("Failed to register tensor id %d failed. Unknown dtype %d",
                 tensor_id, dtype);
  }
  // We move info to avoid a copy.
  data.insert({tensor_id, std::move(info)});
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void dispose_data(int tensor_id) {
  data.erase(tensor_id);

  // Call all disposal callbacks for this tensor id.
  auto disposal_callback_idx = disposal_callbacks.find(tensor_id);
  if (disposal_callback_idx != disposal_callbacks.end()) {
    auto callbacks = disposal_callback_idx->second;
    for (auto dispose_function : callbacks) {
      dispose_function(tensor_id);
    }

    disposal_callbacks.erase(tensor_id);
  }
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void dispose() {
  // We have to create a separate vector of tensor ids because we erase from the
  // map while we're iterating it.
  std::vector<int> tensor_ids_to_dispose;
  for (auto const &element : data) {
    tensor_ids_to_dispose.push_back(element.first);
  }
  for (auto const tensor_id : tensor_ids_to_dispose) {
    dispose_data(tensor_id);
  }

  data.clear();
  disposal_callbacks.clear();
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
