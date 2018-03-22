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
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include "../deps/tensorflow/include/tensorflow/c/eager/c_api.h"
#include "tf_auto_status.h"
#include "tfe_context_env.h"
#include "utils.h"

namespace tfnodejs {

static const std::string CPU_DEVICE_0("cpu:0");

bool IsCPUDevice(std::string& device_name) {
  if (CPU_DEVICE_0.size() > device_name.size()) {
    return false;
  }
  std::transform(device_name.begin(), device_name.end(), device_name.begin(),
                 ::tolower);
  return std::equal(CPU_DEVICE_0.rbegin(), CPU_DEVICE_0.rend(),
                    device_name.rbegin());
}

void Cleanup(napi_env env, void* data, void* hint) {
  TensorHandle* handle = static_cast<TensorHandle*>(data);
  if (handle->handle != nullptr) {
    TFE_DeleteTensorHandle(handle->handle);
    handle->handle = nullptr;
  }
  delete handle;
}

void InitTensorHandle(napi_env env, napi_value wrapped_value) {
  TensorHandle* handle = new TensorHandle();
  handle->handle = nullptr;
  handle->env = env;

  napi_status nstatus =
      napi_wrap(env, wrapped_value, handle, Cleanup, nullptr, nullptr);
  ENSURE_NAPI_OK(env, nstatus);
}

void CopyTensorJSBuffer(napi_env env, napi_value wrapped_value, int64_t* shape,
                        uint32_t shape_length, TF_DataType dtype,
                        napi_value typed_array_value) {
  napi_status nstatus;

  TensorHandle* handle;
  nstatus = napi_unwrap(env, wrapped_value, reinterpret_cast<void**>(&handle));
  ENSURE_NAPI_OK(env, nstatus);

  if (handle->handle != nullptr) {
    // TODO(kreeger): Check to see if the handle can be reused if shape and
    // dtype match.
    TFE_DeleteTensorHandle(handle->handle);
    handle->handle = nullptr;
  }

  napi_typedarray_type array_type;
  size_t array_length;
  void* array_data;
  nstatus =
      napi_get_typedarray_info(env, typed_array_value, &array_type,
                               &array_length, &array_data, nullptr, nullptr);
  ENSURE_NAPI_OK(env, nstatus);

  // Double check the underlying TF_Tensor type matches the supplied
  // typed-array.
  size_t width = 0;
  switch (array_type) {
    case napi_float32_array:
      if (dtype != TF_FLOAT) {
        NAPI_THROW_ERROR(env, "Tensor type does not match Float32Array");
        return;
      }
      width = sizeof(float);
      break;
    case napi_int32_array:
      if (dtype != TF_INT32) {
        NAPI_THROW_ERROR(env, "Tensor type does not match Int32Array");
        return;
      }
      width = sizeof(int32_t);
      break;
    case napi_uint8_array:
      if (dtype != TF_BOOL) {
        NAPI_THROW_ERROR(env, "Tensor type does not match Uint8Array");
        return;
      }
      width = sizeof(uint8_t);
      break;
    default:
      REPORT_UNKNOWN_TYPED_ARRAY_TYPE(env, array_type);
      return;
  }

  // Double check that width matches TF data type size:
  if (width != TF_DataTypeSize(dtype)) {
    NAPI_THROW_ERROR(env, "Byte size of elements differs between JS VM and TF");
    return;
  }

  // Determine the size of the buffer based on the dimensions.
  size_t num_elements = 1;
  for (size_t i = 0; i < shape_length; i++) {
    num_elements *= shape[i];
  }

  // Ensure the shape matches the length of the passed in typed-array.
  if (num_elements != array_length) {
    NAPI_THROW_ERROR(env, "Shape does not match typed-array in bindData()");
    return;
  }

  // Allocate and memcpy JS data to Tensor.
  // TODO(kreeger): Check to see if the Deallocator param can be used to
  // automatically cleanup with JS runtime.
  const size_t byte_size = num_elements * width;
  TF_Tensor* tensor = TF_AllocateTensor(dtype, shape, shape_length, byte_size);
  memcpy(TF_TensorData(tensor), array_data, byte_size);

  TF_AutoStatus tf_status;
  TFE_TensorHandle* tfe_handle = TFE_NewTensorHandle(tensor, tf_status.status);
  ENSURE_TF_OK(env, tf_status);

  // Reference the new TFE_TensorHandle to the wrapped object.
  handle->handle = tfe_handle;

  TF_DeleteTensor(tensor);
}

void GetTensorData(napi_env env, napi_value context_value,
                   napi_value wrapped_value, napi_value* result) {
  napi_status nstatus;

  TFEContextEnv* context_env;
  nstatus =
      napi_unwrap(env, context_value, reinterpret_cast<void**>(&context_env));
  ENSURE_NAPI_OK(env, nstatus);

  if (context_env->context == nullptr) {
    NAPI_THROW_ERROR(env, "Invalid TFE_Context in dataSync()");
    return;
  }

  TensorHandle* handle;
  nstatus = napi_unwrap(env, wrapped_value, reinterpret_cast<void**>(&handle));
  ENSURE_NAPI_OK(env, nstatus);

  if (handle->handle == nullptr) {
    NAPI_THROW_ERROR(env, "Invalid TFE_TensorHandle in dataSync()");
    return;
  }

  // Determine the type of the array
  napi_typedarray_type array_type;
  switch (TFE_TensorHandleDataType(handle->handle)) {
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
      REPORT_UNKNOWN_TF_DATA_TYPE(env,
                                  TFE_TensorHandleDataType(handle->handle));
      return;
  }

  TF_AutoStatus tf_status;

  std::string device_name =
      std::string(TFE_TensorHandleDeviceName(handle->handle, tf_status.status));
  ENSURE_TF_OK(env, tf_status);

  // If the handle is running on a non-CPU device, copy the handle to the device
  // before attempting to read from the tensor buffer.
  bool cleanup_handle = false;
  TFE_TensorHandle* target_handle;
  if (IsCPUDevice(device_name)) {
    target_handle = handle->handle;
  } else {
    target_handle = TFE_TensorHandleCopyToDevice(
        handle->handle, context_env->context, nullptr, tf_status.status);
    ENSURE_TF_OK(env, tf_status);
    cleanup_handle = true;
  }

  TF_Tensor* tensor = TFE_TensorHandleResolve(target_handle, tf_status.status);
  ENSURE_TF_OK(env, tf_status);

  // Determine the length of the array based on the shape of the tensor.
  size_t length = 0;
  uint32_t num_dims = TF_NumDims(tensor);
  for (uint32_t i = 0; i < num_dims; i++) {
    if (i == 0) {
      length = TF_Dim(tensor, i);
    } else {
      length *= TF_Dim(tensor, i);
    }
  }

  void* data = TF_TensorData(tensor);
  size_t byte_length = TF_TensorByteSize(tensor);

  napi_value array_buffer_value;
  nstatus = napi_create_external_arraybuffer(env, data, byte_length, nullptr,
                                             nullptr, &array_buffer_value);
  ENSURE_NAPI_OK(env, nstatus);

  nstatus = napi_create_typedarray(env, array_type, length, array_buffer_value,
                                   0, result);
  ENSURE_NAPI_OK(env, nstatus);

  TF_DeleteTensor(tensor);

  if (cleanup_handle) {
    TFE_DeleteTensorHandle(target_handle);
  }
}

void GetTensorShape(napi_env env, napi_value wrapped_value,
                    napi_value* result) {
  napi_status nstatus;

  TensorHandle* handle;
  nstatus = napi_unwrap(env, wrapped_value, reinterpret_cast<void**>(&handle));
  ENSURE_NAPI_OK(env, nstatus);

  if (handle->handle == nullptr) {
    NAPI_THROW_ERROR(env, "Invalid TFE_TensorHandle used in shape");
    return;
  }

  TF_AutoStatus tf_status;
  uint32_t num_dims = TFE_TensorHandleNumDims(handle->handle, tf_status.status);
  ENSURE_TF_OK(env, tf_status);

  nstatus = napi_create_array_with_length(env, num_dims, result);
  ENSURE_NAPI_OK(env, nstatus);

  for (uint32_t i = 0; i < num_dims; i++) {
    napi_value cur_dim;
    nstatus = napi_create_int64(
        env, TFE_TensorHandleDim(handle->handle, i, tf_status.status),
        &cur_dim);
    ENSURE_TF_OK(env, tf_status);
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

  if (handle->handle == nullptr) {
    NAPI_THROW_ERROR(env, "Invalid TFE_TensorHandle used in dtype");
    return;
  }

  TF_DataType dtype = TFE_TensorHandleDataType(handle->handle);
  nstatus = napi_create_int32(env, dtype, result);
  ENSURE_NAPI_OK(env, nstatus);
}

}  // namespace tfnodejs
