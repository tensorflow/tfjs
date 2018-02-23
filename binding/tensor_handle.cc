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

#include "tensor_handle.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "../deps/tensorflow/include/tensorflow/c/eager/c_api.h"
#include "tf_auto_status.h"
#include "utils.h"

namespace tfnodejs {

void Cleanup(napi_env env, void* data, void* hint) {
  TensorHandle* handle = static_cast<TensorHandle*>(data);
  if (handle->handle != nullptr) {
    TFE_DeleteTensorHandle(handle->handle);
    handle->handle = nullptr;
  }
  if (handle->tensor != nullptr) {
    TF_DeleteTensor(handle->tensor);
    handle->tensor = nullptr;
  }
  delete handle;
}

void InitTensorHandle(napi_env env, napi_value wrapped_value, int64_t* shape,
                      uint32_t shape_length, TF_DataType dtype) {
  TF_AutoStatus tf_status;

  size_t byte_size = 0;
  switch (dtype) {
    case TF_FLOAT:
      byte_size = sizeof(float);
      break;
    case TF_INT32:
      byte_size = sizeof(int32_t);
      break;
    case TF_BOOL:
      byte_size = sizeof(uint8_t);
      break;
    default:
      REPORT_UNKNOWN_TF_DATA_TYPE(dtype);
      break;
  }

  // Determine the size of the buffer based on the dimensions.
  uint32_t buffer_length = 1;
  for (uint32_t i = 0; i < shape_length; i++) {
    buffer_length *= shape[i];
  }

  // Allocate a place holder Tensor. Data will be bound to this later.
  TF_Tensor* tensor =
      TF_AllocateTensor(dtype, shape, shape_length, buffer_length * byte_size);

  TFE_TensorHandle* tfe_handle = TFE_NewTensorHandle(tensor, tf_status.status);
  ENSURE_TF_OK(tf_status);

  // Create underlying wrapper object. It will be reused when data is ready to
  // bind.
  TensorHandle* handle = new TensorHandle();
  handle->tensor = tensor;
  handle->handle = tfe_handle;
  handle->env = env;

  napi_status nstatus =
      napi_wrap(env, wrapped_value, handle, Cleanup, nullptr, nullptr);
  ENSURE_NAPI_OK(env, nstatus);
}

void BindTensorJSBuffer(napi_env env, napi_value wrapped_value,
                        napi_value typed_array_value) {
  napi_status nstatus;

  napi_typedarray_type array_type;
  size_t array_length;
  void* array_data;
  nstatus =
      napi_get_typedarray_info(env, typed_array_value, &array_type,
                               &array_length, &array_data, nullptr, nullptr);
  ENSURE_NAPI_OK(env, nstatus);

  size_t width = 0;
  switch (array_type) {
    case napi_float32_array:
      width = sizeof(float);
      break;
    case napi_int32_array:
      width = sizeof(int32_t);
      break;
    case napi_uint8_array:
      width = sizeof(uint8_t);
      break;
    default:
      REPORT_UNKNOWN_TYPED_ARRAY_TYPE(array_type);
      break;
  }

  TensorHandle* handle;
  nstatus = napi_unwrap(env, wrapped_value, reinterpret_cast<void**>(&handle));
  ENSURE_NAPI_OK(env, nstatus);

  memcpy(TF_TensorData(handle->tensor), array_data, array_length * width);
}

void GetTensorData(napi_env env, napi_value wrapped_value, napi_value* result) {
  napi_status nstatus;

  TensorHandle* handle;
  nstatus = napi_unwrap(env, wrapped_value, reinterpret_cast<void**>(&handle));
  ENSURE_NAPI_OK(env, nstatus);

  // Determine the type of the array
  napi_typedarray_type array_type;
  void* data = TF_TensorData(handle->tensor);
  size_t byte_length = TF_TensorByteSize(handle->tensor);
  switch (TF_TensorType(handle->tensor)) {
    case TF_FLOAT:
      array_type = napi_float32_array;
      break;
    case TF_INT32:
      array_type = napi_int32_array;
      break;
    case TF_BOOL:
      array_type = napi_uint8_array;
      break;
    default:
      REPORT_UNKNOWN_TF_DATA_TYPE(TF_TensorType(handle->tensor));
      break;
  }

  // Determine the length of the array based on the shape of the tensor.
  size_t length = 0;
  uint32_t num_dims = TF_NumDims(handle->tensor);
  for (uint32_t i = 0; i < num_dims; i++) {
    if (i == 0) {
      length = TF_Dim(handle->tensor, i);
    } else {
      length *= TF_Dim(handle->tensor, i);
    }
  }

  napi_value array_buffer_value;
  nstatus = napi_create_external_arraybuffer(env, data, byte_length, nullptr,
                                             nullptr, &array_buffer_value);
  ENSURE_NAPI_OK(env, nstatus);

  nstatus = napi_create_typedarray(env, array_type, length, array_buffer_value,
                                   0, result);
  ENSURE_NAPI_OK(env, nstatus);
}

void GetTensorShape(napi_env env, napi_value wrapped_value,
                    napi_value* result) {
  napi_status nstatus;

  TensorHandle* handle;
  nstatus = napi_unwrap(env, wrapped_value, reinterpret_cast<void**>(&handle));
  ENSURE_NAPI_OK(env, nstatus);

  uint32_t num_dims = TF_NumDims(handle->tensor);
  nstatus = napi_create_array_with_length(env, num_dims, result);
  ENSURE_NAPI_OK(env, nstatus);

  for (uint32_t i = 0; i < num_dims; i++) {
    napi_value cur_dim;
    nstatus = napi_create_int64(env, TF_Dim(handle->tensor, i), &cur_dim);
    ENSURE_NAPI_OK(env, nstatus);

    nstatus = napi_set_element(env, *result, i, cur_dim);
    ENSURE_NAPI_OK(env, nstatus);
  }
}

void GetTensorDtype(napi_env env, napi_value wrapped_value,
                    napi_value* result) {
  napi_status nstatus;

  TensorHandle* handle;
  nstatus = napi_unwrap(env, wrapped_value, reinterpret_cast<void**>(&handle));
  ENSURE_NAPI_OK(env, nstatus);

  TF_DataType dtype = TF_TensorType(handle->tensor);
  nstatus = napi_create_int32(env, dtype, result);
  ENSURE_NAPI_OK(env, nstatus);
}

}  // namespace tfnodejs
