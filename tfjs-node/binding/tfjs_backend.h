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

#ifndef TF_NODEJS_TFJS_BACKEND_H_
#define TF_NODEJS_TFJS_BACKEND_H_

#include <node_api.h>
#include <map>
#include <memory>
#include <string>
#include "tensorflow/c/eager/c_api.h"

namespace tfnodejs {

class TFJSBackend {
 public:
  // Creates, initializes, and returns a TFJSBackend instance. If initialization
  // fails, a nullptr is returned.
  static TFJSBackend* Create(napi_env env);

  // Creates a new Tensor with given shape and data and returns an ID that
  // refernces the new Tensor.
  // - shape_value (number[])
  // - dtype_value (number)
  // - array_value (TypedArray|Array)
  napi_value CreateTensor(napi_env env, napi_value shape_value,
                          napi_value dtype_value, napi_value array_value);

  // Deletes a created Tensor.
  // - tensor_id_value (number)
  void DeleteTensor(napi_env env, napi_value tensor_id_value);

  // Returns a typed-array as a `napi_value` with the data associated with the
  // TF/TFE pointers.
  // - tensor_id_value (number)
  napi_value GetTensorData(napi_env env, napi_value tensor_id_value);

  // Executes a TFE Op and returns an array of objects containing tensor
  // attributes (id, dtype, shape).
  // - op_name_value (string)
  // - op_attr_inputs (array of TFE Op attributes)
  // - input_tensor_ids (array of input tensor IDs)
  // - num_output_values (number)
  napi_value ExecuteOp(napi_env env, napi_value op_name_value,
                       napi_value op_attr_inputs, napi_value input_tensor_ids,
                       napi_value num_output_values);

 private:
  TFJSBackend(napi_env env);
  ~TFJSBackend();

  int32_t InsertHandle(TFE_TensorHandle* tfe_handle);

  TFE_Context* tfe_context_;
  std::map<int32_t, TFE_TensorHandle*> tfe_handle_map_;
  int32_t next_tensor_id_;
  std::string device_name;
};

}  // namespace tfnodejs

#endif  // TF_NODEJS_TFJS_BACKEND_H_
