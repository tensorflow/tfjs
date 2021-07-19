/* Copyright 2019 Google LLC. All Rights Reserved.
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
#include <emscripten/threading.h>
#endif

#include <xnnpack.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/check_macros.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
// Maps a unique tensor id to info about that tensor. The map owns all of its
// entries.
std::unordered_map<size_t, TensorInfo> data;

// Maps a tensor id to a vector of disposal functions registered on that tensor
// id.
std::unordered_map<size_t, std::vector<tfjs::backend::DisposeFunction>>
    disposal_callbacks;
}  // namespace

namespace tfjs {
namespace backend {
const TensorInfo &get_tensor_info(const size_t tensor_id) {
  return data.at(tensor_id);
}

TensorInfo &get_tensor_info_out(const size_t tensor_id) {
  return data.at(tensor_id);
}

size_t xnn_operator_count = 0;

// emscripten_num_logical_cores corresponds to navigator.hardwareConcurrency.
// Many x86-64 processors have 2 threads per core, so we are dividing by 2.
#ifdef __EMSCRIPTEN_PTHREADS__
int num_cores = emscripten_num_logical_cores() / 2;
#else
int num_cores = 1;
#endif

int min_num_threads = 1;
int max_num_threads = 4;
pthreadpool *threadpool = pthreadpool_create(
    std::min(std::max(num_cores, min_num_threads), max_num_threads));

// Registers a disposal callback for a tensor id with a given callback function.
void register_disposal_callback(const size_t tensor_id,
                                const DisposeFunction dispose_fn) {
  if (disposal_callbacks.count(tensor_id) == 0) {
    // We move callbacks to avoid a copy.
    disposal_callbacks.insert({tensor_id, {dispose_fn}});
  } else {
    auto &callbacks = disposal_callbacks[tensor_id];
    callbacks.push_back(dispose_fn);
  }
}

const size_t num_tensors() { return data.size(); }

}  // namespace backend

namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void init() { xnn_initialize(nullptr); }

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void register_tensor(const size_t tensor_id, const size_t size,
                     void *memory_offset) {
  DCHECK(tensor_id > 0,
         "register_tensor: tensor_id must a positive number but got %d.",
         tensor_id);
  DCHECK(data.find(tensor_id) == data.end(),
         "register_tensor: tensor_id %d has already been registered.",
         tensor_id);

  data.emplace(tensor_id, TensorInfo{memory_offset, size});
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void dispose_data(const size_t tensor_id) {
  data.erase(tensor_id);

  // Call all disposal callbacks for this tensor id.
  auto disposal_callback_idx = disposal_callbacks.find(tensor_id);
  if (disposal_callback_idx != disposal_callbacks.end()) {
    auto &callbacks = disposal_callback_idx->second;
    for (auto &dispose_function : callbacks) {
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
  std::vector<size_t> tensor_ids_to_dispose;
  for (const auto &element : data) {
    tensor_ids_to_dispose.push_back(element.first);
  }
  for (const auto tensor_id : tensor_ids_to_dispose) {
    dispose_data(tensor_id);
  }

  data.clear();
  disposal_callbacks.clear();
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
