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

#include "tfjs_backend.h"

#include "tf_auto_tensor.h"
#include "tfe_auto_op.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <set>
#include <string>

namespace tfnodejs {

// Used to hold strings beyond the lifetime of a JS call.
static std::set<std::string> ATTR_NAME_SET;

// Creates a TFE_TensorHandle from a JS typed array.
TFE_TensorHandle *CreateTFE_TensorHandleFromTypedArray(napi_env env,
                                                       int64_t *shape,
                                                       uint32_t shape_length,
                                                       TF_DataType dtype,
                                                       napi_value array_value) {
  napi_status nstatus;
  napi_typedarray_type array_type;
  size_t array_length;
  void *array_data;
  nstatus =
      napi_get_typedarray_info(env, array_value, &array_type, &array_length,
                               &array_data, nullptr, nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  // Double check the underlying TF_Tensor type matches the supplied
  // typed-array.
  size_t width = 0;
  switch (array_type) {
    case napi_float32_array:
      if (dtype != TF_FLOAT) {
        NAPI_THROW_ERROR(env, "Tensor type does not match Float32Array");
        return nullptr;
      }
      width = sizeof(float);
      break;
    case napi_int32_array:
      if (dtype != TF_INT32) {
        NAPI_THROW_ERROR(env, "Tensor type does not match Int32Array");
        return nullptr;
      }
      width = sizeof(int32_t);
      break;
    case napi_uint8_array:
      if (dtype != TF_BOOL) {
        NAPI_THROW_ERROR(env, "Tensor type does not match Uint8Array");
        return nullptr;
      }
      width = sizeof(uint8_t);
      break;
    default:
      REPORT_UNKNOWN_TYPED_ARRAY_TYPE(env, array_type);
      return nullptr;
  }

  // Double check that width matches TF data type size:
  if (width != TF_DataTypeSize(dtype)) {
    NAPI_THROW_ERROR(env,
                     "Byte size of elements differs between JavaScript VM "
                     "(%zu) and TensorFlow (%zu)",
                     width, TF_DataTypeSize(dtype));
    return nullptr;
  }

  // Determine the size of the buffer based on the dimensions.
  size_t num_elements = 1;
  for (size_t i = 0; i < shape_length; i++) {
    num_elements *= shape[i];
  }

  // Ensure the shape matches the length of the passed in typed-array.
  if (num_elements != array_length) {
    NAPI_THROW_ERROR(env,
                     "Shape does not match typed-array in bindData() "
                     "(num_elements=%zu, array_length=%zu)",
                     num_elements, array_length);
    return nullptr;
  }

  // Allocate and memcpy JS data to Tensor.
  const size_t byte_size = num_elements * width;
  TF_AutoTensor tensor(
      TF_AllocateTensor(dtype, shape, shape_length, byte_size));
  memcpy(TF_TensorData(tensor.tensor), array_data, byte_size);

  TF_AutoStatus tf_status;
  TFE_TensorHandle *tfe_tensor_handle =
      TFE_NewTensorHandle(tensor.tensor, tf_status.status);
  ENSURE_TF_OK_RETVAL(env, tf_status, nullptr);

  return tfe_tensor_handle;
}

// Creates a TFE_TensorHandle from a JS array of string values.
TFE_TensorHandle *CreateTFE_TensorHandleFromStringArray(
    napi_env env, int64_t *shape, uint32_t shape_length, TF_DataType dtype,
    napi_value array_value) {
  napi_status nstatus;
  uint32_t array_length;
  nstatus = napi_get_array_length(env, array_value, &array_length);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  size_t offsets_size = array_length * sizeof(uint64_t);
  size_t data_size = offsets_size;
  size_t max_string_length = 0;
  for (uint32_t i = 0; i < array_length; ++i) {
    napi_value cur_value;
    nstatus = napi_get_element(env, array_value, i, &cur_value);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
    ENSURE_VALUE_IS_STRING_RETVAL(env, cur_value, nullptr);

    size_t str_length;
    nstatus =
        napi_get_value_string_utf8(env, cur_value, nullptr, 0, &str_length);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    data_size += TF_StringEncodedSize(str_length);
    max_string_length = std::max(max_string_length, str_length);
  }

  TF_AutoStatus tf_status;
  TF_AutoTensor tensor(
      TF_AllocateTensor(TF_STRING, shape, shape_length, data_size));

  void *tensor_data = TF_TensorData(tensor.tensor);
  uint64_t *offsets = (uint64_t *)tensor_data;

  // Allocate some heap space to work with loading strings to encode with
  // TensorFlow:
  max_string_length++;
  char *buffer = (char *)malloc(sizeof(char *) * max_string_length);

  // Loop past offsets to ensure values fit.
  char *str_data_start = (char *)tensor_data + offsets_size;
  char *cur_str_data = str_data_start;
  for (uint32_t i = 0; i < array_length; ++i) {
    napi_value cur_value;
    nstatus = napi_get_element(env, array_value, i, &cur_value);
    if (nstatus != napi_ok) {
      free(buffer);
      ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
    }

    // Read the current string into the char buffer:
    size_t str_length;
    nstatus = napi_get_value_string_utf8(env, cur_value, buffer,
                                         max_string_length, &str_length);
    if (nstatus != napi_ok) {
      free(buffer);
      ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
    }

    // Append the encoded string into the tensor data chunk:
    size_t encoded_size = TF_StringEncode(buffer, str_length, cur_str_data,
                                          data_size, tf_status.status);
    if (TF_GetCode(tf_status.status) != TF_OK) {
      free(buffer);
      ENSURE_TF_OK_RETVAL(env, tf_status, nullptr);
    }

    offsets[i] = cur_str_data - str_data_start;
    cur_str_data += encoded_size;
  }
  free(buffer);

  TFE_TensorHandle *tfe_tensor_handle =
      TFE_NewTensorHandle(tensor.tensor, tf_status.status);
  ENSURE_TF_OK_RETVAL(env, tf_status, nullptr);
  return tfe_tensor_handle;
}

TFE_TensorHandle *CreateTFE_TensorHandleFromJSValues(napi_env env,
                                                     int64_t *shape,
                                                     uint32_t shape_length,
                                                     TF_DataType dtype,
                                                     napi_value array_value) {
  bool is_typed_array;
  napi_status nstatus = napi_is_typedarray(env, array_value, &is_typed_array);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
  if (is_typed_array) {
    return CreateTFE_TensorHandleFromTypedArray(env, shape, shape_length, dtype,
                                                array_value);
  } else {
    return CreateTFE_TensorHandleFromStringArray(env, shape, shape_length,
                                                 dtype, array_value);
  }
}

TFE_TensorHandle *CopyTFE_TensorHandleToDevice(napi_env env,
                                               const char *device_name,
                                               TFE_TensorHandle *handle,
                                               TFE_Context *tfe_context) {
  TF_AutoStatus tf_status;

  TFE_TensorHandle *new_handle = TFE_TensorHandleCopyToDevice(
      handle, tfe_context, device_name, tf_status.status);
  ENSURE_TF_OK_RETVAL(env, tf_status, nullptr);

  return new_handle;
}

void CopyTFE_TensorHandleDataToTypedArray(napi_env env,
                                          TFE_Context *tfe_context,
                                          TFE_TensorHandle *tfe_tensor_handle,
                                          TF_DataType tensor_data_type,
                                          napi_typedarray_type array_type,
                                          napi_value *result) {
  TF_AutoStatus tf_status;

  TF_AutoTensor tensor(
      TFE_TensorHandleResolve(tfe_tensor_handle, tf_status.status));
  ENSURE_TF_OK(env, tf_status);

  // Determine the length of the array based on the shape of the tensor.
  size_t num_elements = GetTensorNumElements(tensor.tensor);

  if (tensor_data_type == TF_COMPLEX64) {
    // Dimension length will be double for Complex 64.
    num_elements *= 2;
  }

  size_t byte_length = TF_TensorByteSize(tensor.tensor);

  napi_value array_buffer_value;
  void *array_buffer_data;
  napi_status nstatus;
  nstatus = napi_create_arraybuffer(env, byte_length, &array_buffer_data,
                                    &array_buffer_value);
  ENSURE_NAPI_OK(env, nstatus);

  // TFE_TensorHandleResolve can use a shared data pointer, memcpy() the current
  // value to the newly allocated NAPI buffer.
  memcpy(array_buffer_data, TF_TensorData(tensor.tensor), byte_length);

  nstatus = napi_create_typedarray(env, array_type, num_elements,
                                   array_buffer_value, 0, result);
  ENSURE_NAPI_OK(env, nstatus);
}

void CopyTFE_TensorHandleDataToStringArray(napi_env env,
                                           TFE_Context *tfe_context,
                                           TFE_TensorHandle *tfe_tensor_handle,
                                           napi_value *result) {
  TF_AutoStatus tf_status;

  TF_AutoTensor tensor(
      TFE_TensorHandleResolve(tfe_tensor_handle, tf_status.status));
  ENSURE_TF_OK(env, tf_status);

  if (TF_TensorType(tensor.tensor) != TF_STRING) {
    NAPI_THROW_ERROR(env, "Tensor is not of type TF_STRING");
    return;
  }

  void *tensor_data = TF_TensorData(tensor.tensor);
  ENSURE_VALUE_IS_NOT_NULL(env, tensor_data);

  size_t byte_length = TF_TensorByteSize(tensor.tensor);
  const char *limit = static_cast<const char *>(tensor_data) + byte_length;

  size_t num_elements = GetTensorNumElements(tensor.tensor);

  // String values are stored in offsets.
  const uint64_t *offsets = static_cast<const uint64_t *>(tensor_data);
  const size_t offsets_size = sizeof(uint64_t) * num_elements;

  // Skip passed the offsets and find the first string:
  const char *data = static_cast<const char *>(tensor_data) + offsets_size;

  TF_AutoStatus status;

  // Create a JS string to stash strings into
  napi_status nstatus;
  nstatus = napi_create_array_with_length(env, num_elements, result);

  const size_t expected_tensor_size =
      (limit - static_cast<const char *>(tensor_data));
  if (expected_tensor_size != byte_length) {
    NAPI_THROW_ERROR(env,
                     "Invalid/corrupt TF_STRING tensor. Expected size: %zu, "
                     "byte_length: %zu",
                     expected_tensor_size, byte_length);
    return;
  }

  for (uint64_t i = 0; i < num_elements; i++) {
    const char *start = data + offsets[i];
    const char *str_ptr = nullptr;
    size_t str_len = 0;

    TF_StringDecode(start, limit - start, &str_ptr, &str_len, status.status);
    ENSURE_TF_OK(env, tf_status);

    napi_value str_value;
    nstatus = napi_create_string_utf8(env, str_ptr, str_len, &str_value);
    ENSURE_NAPI_OK(env, nstatus);

    nstatus = napi_set_element(env, *result, i, str_value);
    ENSURE_NAPI_OK(env, nstatus);
  }
}

// Handles converting the stored TF_Tensor data into the correct JS value.
void CopyTFE_TensorHandleDataToJSData(napi_env env, TFE_Context *tfe_context,
                                      TFE_TensorHandle *tfe_tensor_handle,
                                      napi_value *result) {
  if (tfe_context == nullptr) {
    NAPI_THROW_ERROR(env, "Invalid TFE_Context");
    return;
  }
  if (tfe_tensor_handle == nullptr) {
    NAPI_THROW_ERROR(env, "Invalid TFE_TensorHandle");
    return;
  }

  // Determine the type of the array
  napi_typedarray_type typed_array_type;
  bool is_string = false;
  TF_DataType tensor_data_type = TFE_TensorHandleDataType(tfe_tensor_handle);
  switch (tensor_data_type) {
    case TF_COMPLEX64:
    case TF_FLOAT:
      typed_array_type = napi_float32_array;
      break;
    case TF_INT32:
      typed_array_type = napi_int32_array;
      break;
    case TF_BOOL:
      typed_array_type = napi_uint8_array;
      break;
    case TF_STRING:
      is_string = true;
      break;
    default:
      REPORT_UNKNOWN_TF_DATA_TYPE(env,
                                  TFE_TensorHandleDataType(tfe_tensor_handle));
      return;
  }

  if (is_string) {
    CopyTFE_TensorHandleDataToStringArray(env, tfe_context, tfe_tensor_handle,
                                          result);
  } else {
    CopyTFE_TensorHandleDataToTypedArray(env, tfe_context, tfe_tensor_handle,
                                         tensor_data_type, typed_array_type,
                                         result);
  }
}

void GetTFE_TensorHandleShape(napi_env env, TFE_TensorHandle *handle,
                              napi_value *result) {
  napi_status nstatus;

  TF_AutoStatus tf_status;
  uint32_t num_dims = TFE_TensorHandleNumDims(handle, tf_status.status);
  ENSURE_TF_OK(env, tf_status);

  if (num_dims == 0) {
    nstatus = napi_create_array_with_length(env, 0, result);
    ENSURE_NAPI_OK(env, nstatus);
  } else {
    nstatus = napi_create_array_with_length(env, num_dims, result);
    ENSURE_NAPI_OK(env, nstatus);

    for (uint32_t i = 0; i < num_dims; i++) {
      napi_value cur_dim;
      nstatus = napi_create_int64(
          env, TFE_TensorHandleDim(handle, i, tf_status.status), &cur_dim);
      ENSURE_TF_OK(env, tf_status);
      ENSURE_NAPI_OK(env, nstatus);

      nstatus = napi_set_element(env, *result, i, cur_dim);
      ENSURE_NAPI_OK(env, nstatus);
    }
  }
}

inline bool IsArray(napi_env env, napi_status &nstatus, napi_value *val) {
  bool is_array;
  nstatus = napi_is_array(env, *val, &is_array);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, false);
  return is_array;
}

void GetTFE_TensorHandleType(napi_env env, TFE_TensorHandle *handle,
                             napi_value *result) {
  napi_status nstatus;

  TF_DataType dtype = TFE_TensorHandleDataType(handle);
  nstatus = napi_create_int32(env, dtype, result);
  ENSURE_NAPI_OK(env, nstatus);
}

void AssignOpAttr(napi_env env, TFE_Op *tfe_op, napi_value attr_value) {
  napi_status nstatus;

  napi_value attr_name_value;
  nstatus = napi_get_named_property(env, attr_value, "name", &attr_name_value);
  ENSURE_NAPI_OK(env, nstatus);

  std::string attr_name_string;
  nstatus = GetStringParam(env, attr_name_value, attr_name_string);
  ENSURE_NAPI_OK(env, nstatus);

  // OpAttr will be used beyond the scope of this function call. Stash ops in a
  // set for re-use instead of dynamically reallocating strings for operations.
  const char *attr_name =
      ATTR_NAME_SET.insert(attr_name_string.c_str()).first->c_str();

  napi_value attr_type_value;
  nstatus = napi_get_named_property(env, attr_value, "type", &attr_type_value);
  ENSURE_NAPI_OK(env, nstatus);

  TF_AttrType tf_attr_type;
  nstatus = napi_get_value_int32(env, attr_type_value,
                                 reinterpret_cast<int32_t *>(&tf_attr_type));
  ENSURE_NAPI_OK(env, nstatus);

  napi_value js_value;
  nstatus = napi_get_named_property(env, attr_value, "value", &js_value);
  ENSURE_NAPI_OK(env, nstatus);

  switch (tf_attr_type) {
    case TF_ATTR_STRING: {
      // NOTE: String attribute values do not have to be utf8 encoded strings
      // (could be arbitrary byte sequences).
      std::string str_value;
      nstatus = GetStringParam(env, js_value, str_value);
      ENSURE_NAPI_OK(env, nstatus);

      TFE_OpSetAttrString(tfe_op, attr_name, str_value.c_str(),
                          str_value.size());
      break;
    }

    case TF_ATTR_INT: {
      if (IsArray(env, nstatus, &js_value)) {
        uint32_t length;
        nstatus = napi_get_array_length(env, js_value, &length);
        ENSURE_NAPI_OK(env, nstatus);
        std::unique_ptr<int64_t[]> data(new int64_t[length]);
        for (uint32_t i = 0; i < length; ++i) {
          napi_value element;
          nstatus = napi_get_element(env, js_value, i, &element);
          ENSURE_NAPI_OK(env, nstatus);
          int32_t value;
          nstatus = napi_get_value_int32(env, element, &value);
          ENSURE_NAPI_OK(env, nstatus);
          data[i] = value;
        }
        TFE_OpSetAttrIntList(tfe_op, attr_name, data.get(),
                             static_cast<int>(length));
      } else {
        int64_t value;
        nstatus = napi_get_value_int64(env, js_value, &value);
        ENSURE_NAPI_OK(env, nstatus);

        TFE_OpSetAttrInt(tfe_op, attr_name, value);
      }
      break;
    }

    case TF_ATTR_FLOAT: {
      if (IsArray(env, nstatus, &js_value)) {
        uint32_t length;
        nstatus = napi_get_array_length(env, js_value, &length);
        ENSURE_NAPI_OK(env, nstatus);
        std::unique_ptr<float[]> data(new float[length]);
        for (uint32_t i = 0; i < length; ++i) {
          napi_value element;
          nstatus = napi_get_element(env, js_value, i, &element);
          ENSURE_NAPI_OK(env, nstatus);
          double value;
          nstatus = napi_get_value_double(env, element, &value);
          ENSURE_NAPI_OK(env, nstatus);
          data[i] = static_cast<float>(value);
        }
        TFE_OpSetAttrFloatList(tfe_op, attr_name, data.get(),
                               static_cast<int>(length));
      } else {
        double value;
        nstatus = napi_get_value_double(env, js_value, &value);
        ENSURE_NAPI_OK(env, nstatus);
        TFE_OpSetAttrFloat(tfe_op, attr_name, static_cast<float>(value));
      }
      break;
    }

    case TF_ATTR_BOOL: {
      if (IsArray(env, nstatus, &js_value)) {
        uint32_t length;
        nstatus = napi_get_array_length(env, js_value, &length);
        ENSURE_NAPI_OK(env, nstatus);
        std::unique_ptr<unsigned char[]> data(new unsigned char[length]);
        for (uint32_t i = 0; i < length; ++i) {
          napi_value element;
          nstatus = napi_get_element(env, js_value, i, &element);
          ENSURE_NAPI_OK(env, nstatus);
          bool value;
          nstatus = napi_get_value_bool(env, element, &value);
          ENSURE_NAPI_OK(env, nstatus);
          data[i] = value ? 1 : 0;
        }
        TFE_OpSetAttrBoolList(tfe_op, attr_name, data.get(),
                              static_cast<int>(length));
      } else {
        bool value;
        nstatus = napi_get_value_bool(env, js_value, &value);
        ENSURE_NAPI_OK(env, nstatus);
        TFE_OpSetAttrBool(tfe_op, attr_name, value ? 1 : 0);
      }
      break;
    }

    case TF_ATTR_TYPE: {
      TF_DataType tf_data_type;
      nstatus = napi_get_value_int32(
          env, js_value, reinterpret_cast<int32_t *>(&tf_data_type));
      ENSURE_NAPI_OK(env, nstatus);

      TFE_OpSetAttrType(tfe_op, attr_name, tf_data_type);
      break;
    }

    case TF_ATTR_SHAPE: {
      std::vector<int64_t> shape_vector;
      ExtractArrayShape(env, js_value, &shape_vector);

      TF_AutoStatus tf_status;
      TFE_OpSetAttrShape(tfe_op, attr_name, shape_vector.data(),
                         shape_vector.size(), tf_status.status);
      ENSURE_TF_OK(env, tf_status);
      break;
    }

    default:
      REPORT_UNKNOWN_TF_ATTR_TYPE(env, tf_attr_type);
      break;
  }
}

TFJSBackend::TFJSBackend(napi_env env) : next_tensor_id_(0) {
  TF_AutoStatus tf_status;
  TFE_ContextOptions *tfe_options = TFE_NewContextOptions();
  tfe_context_ = TFE_NewContext(tfe_options, tf_status.status);
  if (TF_GetCode(tf_status.status) != TF_OK) {
    NAPI_THROW_ERROR(env, "Exception creating TFE_Context");
  }

  TFE_DeleteContextOptions(tfe_options);

  TF_DeviceList *device_list =
      TFE_ContextListDevices(tfe_context_, tf_status.status);
  if (TF_GetCode(tf_status.status) != TF_OK) {
    NAPI_THROW_ERROR(env, "Exception creating TFE_Context");
  }

  const int num_devices = TF_DeviceListCount(device_list);
  for (int i = 0; i < num_devices; i++) {
    // Always use the last device (CPU is listed first).
    // TODO(kreeger): Add better support for this in the future through the JS
    // API. https://github.com/tensorflow/tfjs/issues/320
    device_name =
        std::string(TF_DeviceListName(device_list, i, tf_status.status));
  }
  TF_DeleteDeviceList(device_list);
}

TFJSBackend::~TFJSBackend() {
  for (auto &kv : tfe_handle_map_) {
    TFE_DeleteTensorHandle(kv.second);
  }
  if (tfe_context_ != nullptr) {
    TFE_DeleteContext(tfe_context_);
  }
}

TFJSBackend *TFJSBackend::Create(napi_env env) { return new TFJSBackend(env); }

int32_t TFJSBackend::InsertHandle(TFE_TensorHandle *tfe_handle) {
  return tfe_handle_map_.insert(std::make_pair(next_tensor_id_++, tfe_handle))
      .first->first;
}

napi_value TFJSBackend::CreateTensor(napi_env env, napi_value shape_value,
                                     napi_value dtype_value,
                                     napi_value array_value) {
  napi_status nstatus;

  std::vector<int64_t> shape_vector;
  ExtractArrayShape(env, shape_value, &shape_vector);
  // Check to see if an exception exists, if so return a failure.
  if (IsExceptionPending(env)) {
    return nullptr;
  }

  int32_t dtype_int32;
  nstatus = napi_get_value_int32(env, dtype_value, &dtype_int32);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  TFE_TensorHandle *tfe_handle = CreateTFE_TensorHandleFromJSValues(
      env, shape_vector.data(), shape_vector.size(),
      static_cast<TF_DataType>(dtype_int32), array_value);

  // Check to see if an exception exists, if so return a failure.
  if (IsExceptionPending(env)) {
    return nullptr;
  }

  // Copy non-int32 tensors to a device. Most GPU kernels expect to have int32
  // tensors in host memory.
  if (dtype_int32 != TF_INT32) {
    // Note that this is a shallow copy and will share the underlying buffer
    // if copying to the same device.
    TFE_TensorHandle *new_handle = CopyTFE_TensorHandleToDevice(
        env, device_name.c_str(), tfe_handle, tfe_context_);

    TFE_DeleteTensorHandle(tfe_handle);
    tfe_handle = new_handle;
  }

  napi_value output_tensor_id;
  nstatus = napi_create_int32(env, InsertHandle(tfe_handle), &output_tensor_id);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
  return output_tensor_id;
}

void TFJSBackend::DeleteTensor(napi_env env, napi_value tensor_id_value) {
  int32_t tensor_id;
  ENSURE_NAPI_OK(env, napi_get_value_int32(env, tensor_id_value, &tensor_id));

  auto tensor_entry = tfe_handle_map_.find(tensor_id);
  if (tensor_entry == tfe_handle_map_.end()) {
    NAPI_THROW_ERROR(env,
                     "Delete called on a Tensor not referenced (tensor_id: %d)",
                     tensor_id);
    return;
  }

  TFE_DeleteTensorHandle(tensor_entry->second);
  tfe_handle_map_.erase(tensor_entry);
}

napi_value TFJSBackend::GetTensorData(napi_env env,
                                      napi_value tensor_id_value) {
  int32_t tensor_id;
  ENSURE_NAPI_OK_RETVAL(
      env, napi_get_value_int32(env, tensor_id_value, &tensor_id), nullptr);

  auto tensor_entry = tfe_handle_map_.find(tensor_id);
  if (tensor_entry == tfe_handle_map_.end()) {
    NAPI_THROW_ERROR(
        env, "Get data called on a Tensor not referenced (tensor_id: %d)",
        tensor_id);
    return nullptr;
  }

  napi_value js_value;
  CopyTFE_TensorHandleDataToJSData(env, tfe_context_, tensor_entry->second,
                                   &js_value);
  return js_value;
}

napi_value TFJSBackend::ExecuteOp(napi_env env, napi_value op_name_value,
                                  napi_value op_attr_inputs,
                                  napi_value input_tensor_ids,
                                  napi_value num_output_values) {
  napi_status nstatus;

  std::string op_name;
  nstatus = GetStringParam(env, op_name_value, op_name);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  TF_AutoStatus tf_status;
  TFE_AutoOp tfe_op(TFE_NewOp(tfe_context_, op_name.c_str(), tf_status.status));
  ENSURE_TF_OK_RETVAL(env, tf_status, nullptr);

  uint32_t num_input_ids;
  nstatus = napi_get_array_length(env, input_tensor_ids, &num_input_ids);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  for (uint32_t i = 0; i < num_input_ids; i++) {
    napi_value cur_input_id;
    nstatus = napi_get_element(env, input_tensor_ids, i, &cur_input_id);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    int32_t cur_input_tensor_id;
    nstatus = napi_get_value_int32(env, cur_input_id, &cur_input_tensor_id);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    auto input_tensor_entry = tfe_handle_map_.find(cur_input_tensor_id);
    if (input_tensor_entry == tfe_handle_map_.end()) {
      NAPI_THROW_ERROR(env, "Input Tensor ID not referenced (tensor_id: %d)",
                       cur_input_tensor_id);
      return nullptr;
    }

    TFE_OpAddInput(tfe_op.op, input_tensor_entry->second, tf_status.status);
    ENSURE_TF_OK_RETVAL(env, tf_status, nullptr);
  }

  uint32_t op_attrs_length;
  nstatus = napi_get_array_length(env, op_attr_inputs, &op_attrs_length);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  for (uint32_t i = 0; i < op_attrs_length; i++) {
    napi_value cur_op_attr;
    nstatus = napi_get_element(env, op_attr_inputs, i, &cur_op_attr);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    AssignOpAttr(env, tfe_op.op, cur_op_attr);

    // Check to see if an exception exists, if so return a failure.
    if (IsExceptionPending(env)) {
      return nullptr;
    }
  }

  int32_t num_outputs;
  nstatus = napi_get_value_int32(env, num_output_values, &num_outputs);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  // Push `nullptr` to get a valid pointer in the call to `TFE_Execute()` below.
  std::vector<TFE_TensorHandle *> result_handles(num_outputs, nullptr);

  int size = result_handles.size();
  TFE_Execute(tfe_op.op, result_handles.data(), &size, tf_status.status);
  ENSURE_TF_OK_RETVAL(env, tf_status, nullptr);

  napi_value output_tensor_infos;
  nstatus = napi_create_array_with_length(env, size, &output_tensor_infos);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  for (int32_t i = 0; i < num_outputs; i++) {
    // Output tensor info object:
    napi_value tensor_info_value;
    nstatus = napi_create_object(env, &tensor_info_value);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    TFE_TensorHandle *handle = result_handles[i];

    // Output tensor ID:
    napi_value output_tensor_id_value;
    nstatus =
        napi_create_int32(env, InsertHandle(handle), &output_tensor_id_value);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    nstatus = napi_set_named_property(env, tensor_info_value, "id",
                                      output_tensor_id_value);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    // Output tensor shape:
    napi_value shape_value;
    GetTFE_TensorHandleShape(env, handle, &shape_value);

    nstatus =
        napi_set_named_property(env, tensor_info_value, "shape", shape_value);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    // Output tensor dtype:
    napi_value type_value;
    GetTFE_TensorHandleType(env, handle, &type_value);

    nstatus =
        napi_set_named_property(env, tensor_info_value, "dtype", type_value);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    // Push into output array
    nstatus = napi_set_element(env, output_tensor_infos, i, tensor_info_value);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
  }

  return output_tensor_infos;
}

}  // namespace tfnodejs
