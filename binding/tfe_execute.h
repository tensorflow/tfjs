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

#ifndef TF_NODEJS_TFE_EXECUTE_H_
#define TF_NODEJS_TFE_EXECUTE_H_

#include <node_api.h>

namespace tfnodejs {

// Executes a TFE Op based on the name, inputs, attributes, and output.
void ExecuteOp(napi_env env, napi_value context, const char* opName,
               napi_value op_attr_inputs, napi_value inputs,
               napi_value output_tensor);
}  // namespace tfnodejs

#endif  // TF_NODEJS_TFE_EXECUTE_H_