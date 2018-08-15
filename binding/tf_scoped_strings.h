/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
 * =============================================================================
 */

#ifndef TF_NODEJS_TF_SCOPED_STRINGS_H_
#define TF_NODEJS_TF_SCOPED_STRINGS_H_

#include "utils.h"

#include <node_api.h>
#include <memory>
#include <string>
#include <vector>

namespace tfnodejs {

// Manges a vector of heap-allocated strings for the life-span of the object.
// TODO(kreeger): Drop this class when 1.11 TensorFlow is released:
// https://github.com/tensorflow/tfjs-node/pull/146#discussion_r210160129
class TF_ScopedStrings {
 public:
  // Returns a string or nullptr pointer for the underlying JS object.
  std::string* GetString(napi_env env, napi_value js_value) {
    size_t size;
    char buffer[NAPI_STRING_SIZE];
    napi_status nstatus = napi_get_value_string_utf8(env, js_value, buffer,
                                                     NAPI_STRING_SIZE, &size);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
    string_refs_.push_back(
        std::unique_ptr<std::string>(new std::string(buffer, size)));
    return string_refs_.back().get();
  }

 private:
  std::vector<std::unique_ptr<std::string>> string_refs_;
};

}  // namespace tfnodejs

#endif  // TF_NODEJS_TF_SCOPED_STRINGS_H_
