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

#ifndef TF_NODEJS_TENSOR_BINDING_H_
#define TF_NODEJS_TENSOR_BINDING_H_

#include <node_api.h>
#include "../deps/tensorflow/include/tensorflow/c/eager/c_api.h"

namespace tfnodejs {

struct WrappedTensorHandle {
  TFE_TensorHandle* handle;
  napi_env env;
};

// Initializes a `TensorHandle` object with the napi_env set, but null TF/TFE
// pointers
void InitTensorHandle(napi_env env, napi_value wrapped_value);

// Copies a JS typed-array to the wrapped TF/TFE pointers.
void CopyTensorJSBuffer(napi_env env, napi_value wrapped_value, int64_t* shape,
                        uint32_t shape_length, TF_DataType dtype,
                        napi_value typed_array_value);

// Returns a typed-array as a `napi_value` with the data associated with the
// TF/TFE pointers.
void GetTensorData(napi_env env, napi_value context_value,
                   napi_value wrapped_value, napi_value* result);

// Returns an array as a `napi_value` with shape of the Tensor.
void GetTensorShape(napi_env env, napi_value wrapped_value, napi_value* result);

// Returns a type as a `napi_value` with the type of the Tensor.
void GetTensorDtype(napi_env env, napi_value wrapped_value, napi_value* result);

}  // namespace tfnodejs

#endif  // TF_NODEJS_TENSOR_BINDING_H_
