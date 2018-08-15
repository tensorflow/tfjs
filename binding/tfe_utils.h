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

#ifndef TF_NODEJS_TFE_UTILS_H_
#define TF_NODEJS_TFE_UTILS_H_

#include "tf_scoped_strings.h"

#include <node_api.h>
#include "../deps/include/tensorflow/c/eager/c_api.h"

namespace tfnodejs {

// Creates a TFE_TensorHandle from a typed array.
TFE_TensorHandle* CreateTFE_TensorHandleFromTypedArray(
    napi_env env, int64_t* shape, uint32_t shape_length, TF_DataType dtype,
    napi_value typed_array_value);

// Copies a handle to a device
TFE_TensorHandle* CopyTFE_TensorHandleToDevice(napi_env env,
                                               const char* device_name,
                                               TFE_TensorHandle* handle,
                                               TFE_Context* tfe_context);

// Returns a typed-array as a `napi_value` with the data associated with the
// TF/TFE pointers.
void CopyTFE_TensorHandleDataToTypedArray(napi_env env,
                                          TFE_Context* tfe_context,
                                          TFE_TensorHandle* tfe_tensor_handle,
                                          napi_value* result);

// Returns an array as a `napi_value` with shape of the Tensor.
void GetTFE_TensorHandleShape(napi_env env, TFE_TensorHandle* handle,
                              napi_value* result);

// Returns a type as a `napi_value` with the type of the Tensor.
void GetTFE_TensorHandleType(napi_env env, TFE_TensorHandle* handle,
                             napi_value* result);

// Assigns attributes to an Op from a given list of inputs attributes.
void AssignOpAttr(napi_env env, TFE_Op* tfe_op, napi_value attr_value,
                  tfnodejs::TF_ScopedStrings* scoped_strings);

}  // namespace tfnodejs

#endif  // TF_NODEJS_TFE_UTILS_H_
