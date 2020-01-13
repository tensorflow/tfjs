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

#include <algorithm>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include "napi_auto_ref.h"
#include "tf_auto_tensor.h"
#include "tfe_auto_op.h"
#include "utils.h"

namespace tfnodejs {

// Used to hold strings beyond the lifetime of a JS call.
static std::set<std::string> ATTR_NAME_SET;

// Callback to cleanup extra reference count for shared V8/TF tensor memory:
static void DeallocTensor(void *data, size_t len, void *arg) {
  NapiAutoRef *auto_ref = static_cast<NapiAutoRef *>(arg);
  if (!auto_ref) {
#if DEBUG
    fprintf(stderr, "Invalid NapiAutoRef reference passed to V8 cleanup\n");
#endif
    return;
  }
  if (auto_ref->Cleanup() != napi_ok) {
#if DEBUG
    fprintf(stderr, "Exception cleaning up napi_ref instance\n");
#endif
  }
  delete auto_ref;
}

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
      if (dtype != TF_INT32 && dtype != TF_INT64) {
        // Currently, both int32- and int64-type Tensors are represented
        // as Int32Arrays in JavaScript. See int64_tensors.ts for details
        // about the latter.
        NAPI_THROW_ERROR(env, "Tensor type does not match Int32Array");
        return nullptr;
      }
      width = sizeof(int32_t);
      break;
    case napi_uint8_array:
      if (dtype != TF_BOOL && dtype != TF_UINT8) {
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
  if (dtype == TF_INT64) {
    // Currently, int64-type Tensors are represented as Int32Arrays. So the
    // logic for comparing the byte size of the typed-array representation and
    // the byte size of the tensor dtype needs to be special-cased for int64.
    if (width * 2 != TF_DataTypeSize(dtype)) {
      NAPI_THROW_ERROR(
          env,
          "Byte size of elements differs between JavaScript VM "
          "(%zu * 2 = %zu) and TensorFlow (%zu) for int64-type tensor",
          width, width * 2, TF_DataTypeSize(dtype));
      return nullptr;
    }
  } else {
    if (width != TF_DataTypeSize(dtype)) {
      NAPI_THROW_ERROR(env,
                       "Byte size of elements differs between JavaScript VM "
                       "(%zu) and TensorFlow (%zu)",
                       width, TF_DataTypeSize(dtype));
      return nullptr;
    }
  }

  // Determine the size of the buffer based on the dimensions.
  size_t num_elements = 1;
  for (size_t i = 0; i < shape_length; i++) {
    num_elements *= shape[i];
  }

  // Ensure the shape matches the length of the passed in typed-array.
  if (dtype == TF_INT64) {
    // Currently, int64-type Tensors are represented as Int32Arrays.
    // To represent a int64-type Tensor of `n` elements, an Int32Array of
    // length `2 * n` is requried. This is why the length-match checking
    // logic is special-cased for int64.
    if (array_length != num_elements * 2) {
      NAPI_THROW_ERROR(
          env,
          "Shape does not match two times typed-array in bindData() "
          "(num_elements * 2 = %zu, array_length=%zu) for int64 data type",
          num_elements * 2, array_length);
      return nullptr;
    }
  } else {
    if (num_elements != array_length) {
      NAPI_THROW_ERROR(env,
                       "Shape does not match typed-array in bindData() "
                       "(num_elements=%zu, array_length=%zu)",
                       num_elements, array_length);
      return nullptr;
    }
  }

  // Sharing V8 memory with the underlying TensorFlow tensor requires adding an
  // additional refcount. When the Tensor is deleted, the refcount will be
  // reduced in the callback helper.
  NapiAutoRef *auto_ref = new NapiAutoRef();
  nstatus = auto_ref->Init(env, array_value);
  if (nstatus != napi_ok) {
    delete auto_ref;
  }
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  // Currently, int64-type Tensors are represented as Int32Arrays.
  // So the logic for comparing the byte size of the typed-array representation
  // and the byte size of the tensor dtype needs to be special-cased for int64.
  const size_t byte_size =
      dtype == TF_INT64 ? num_elements * width * 2 : num_elements * width;

  TF_AutoTensor tensor(TF_NewTensor(dtype, shape, shape_length, array_data,
                                    byte_size, DeallocTensor, auto_ref));

  TF_AutoStatus tf_status;
  TFE_TensorHandle *tfe_tensor_handle =
      TFE_NewTensorHandle(tensor.tensor, tf_status.status);
  if (TF_GetCode(tf_status.status) != TF_OK) {
    delete auto_ref;
    TFE_DeleteTensorHandle(tfe_tensor_handle);
  }
  ENSURE_TF_OK_RETVAL(env, tf_status, nullptr);

  return tfe_tensor_handle;
}

// Creates a TFE_TensorHandle from a JS array of Uint8Array values.
TFE_TensorHandle *CreateTFE_TensorHandleFromStringArray(
    napi_env env, int64_t *shape, uint32_t shape_length, TF_DataType dtype,
    napi_value array_value) {
  napi_status nstatus;

  uint32_t array_length;
  nstatus = napi_get_array_length(env, array_value, &array_length);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  size_t offsets_size = array_length * sizeof(uint64_t);
  size_t data_size = offsets_size;

  for (uint32_t i = 0; i < array_length; ++i) {
    napi_value cur_value;
    nstatus = napi_get_element(env, array_value, i, &cur_value);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
    ENSURE_VALUE_IS_TYPED_ARRAY_RETVAL(env, cur_value, nullptr);

    size_t cur_array_length;
    napi_typedarray_type array_type;
    nstatus =
        napi_get_typedarray_info(env, cur_value, &array_type, &cur_array_length,
                                 nullptr, nullptr, nullptr);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    // Only Uint8 typed arrays are supported.
    if (array_type != napi_uint8_array) {
      NAPI_THROW_ERROR(env, "Unsupported array type - expecting Uint8Array");
      return nullptr;
    }

    data_size += TF_StringEncodedSize(cur_array_length);
  }

  TF_AutoStatus tf_status;
  TF_AutoTensor tensor(
      TF_AllocateTensor(TF_STRING, shape, shape_length, data_size));

  void *tensor_data = TF_TensorData(tensor.tensor);
  uint64_t *offsets = (uint64_t *)tensor_data;

  char *str_data_start = (char *)tensor_data + offsets_size;
  char *cur_str_data = str_data_start;

  for (uint32_t i = 0; i < array_length; ++i) {
    napi_value cur_value;
    nstatus = napi_get_element(env, array_value, i, &cur_value);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    size_t cur_array_length;
    void *buffer = nullptr;
    nstatus = napi_get_typedarray_info(
        env, cur_value, nullptr, &cur_array_length, &buffer, nullptr, nullptr);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    size_t encoded_size =
        TF_StringEncode(reinterpret_cast<char *>(buffer), cur_array_length,
                        cur_str_data, data_size, tf_status.status);
    ENSURE_TF_OK_RETVAL(env, tf_status, nullptr);

    offsets[i] = cur_str_data - str_data_start;
    cur_str_data += encoded_size;
  }

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

  // TFE_TensorHandleResolve can use a shared data pointer, memcpy() the
  // current value to the newly allocated NAPI buffer.
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

    napi_value array_buffer_value;
    void *array_buffer_data;
    nstatus = napi_create_arraybuffer(env, str_len, &array_buffer_data,
                                      &array_buffer_value);
    ENSURE_NAPI_OK(env, nstatus);

    // TF_StringDecode returns a const char pointer that can not be used
    // directly because of const rules in napi_create_arraybuffer. Simply memcpy
    // the buffers here.
    memcpy(array_buffer_data, str_ptr, str_len);

    napi_value typed_array_value;
    nstatus = napi_create_typedarray(env, napi_uint8_array, str_len,
                                     array_buffer_value, 0, &typed_array_value);
    ENSURE_NAPI_OK(env, nstatus);

    nstatus = napi_set_element(env, *result, i, typed_array_value);
    ENSURE_NAPI_OK(env, nstatus);
  }
}

void CopyTFE_TensorHandleDataToResourceArray(
    napi_env env, TFE_Context *tfe_context, TFE_TensorHandle *tfe_tensor_handle,
    napi_value *result) {
  TF_AutoStatus tf_status;

  TF_AutoTensor tensor(
      TFE_TensorHandleResolve(tfe_tensor_handle, tf_status.status));
  ENSURE_TF_OK(env, tf_status);

  if (TF_TensorType(tensor.tensor) != TF_RESOURCE) {
    NAPI_THROW_ERROR(env, "Tensor is not of type TF_RESOURCE");
    return;
  }

  void *tensor_data = TF_TensorData(tensor.tensor);
  ENSURE_VALUE_IS_NOT_NULL(env, tensor_data);

  size_t num_elements = GetTensorNumElements(tensor.tensor);
  if (num_elements != 1) {
    NAPI_THROW_ERROR(env,
                     "For DT_RESOURCE tensors, Node.js binding currently "
                     "supports only exactly 1 element, but encountered "
                     "DT_RESOURCE tensor with %zu elements.",
                     num_elements);
  }

  TF_AutoStatus status;

  // Create a JS string to stash the resouce handle into.
  napi_status nstatus;
  size_t byte_length = TF_TensorByteSize(tensor.tensor);
  nstatus = napi_create_array_with_length(env, byte_length, result);
  ENSURE_NAPI_OK(env, nstatus);

  napi_value array_buffer_value;
  void *array_buffer_data = nullptr;
  nstatus = napi_create_arraybuffer(env, byte_length, &array_buffer_data,
                                    &array_buffer_value);
  ENSURE_NAPI_OK(env, nstatus);

  // TFE_TensorHandleResolve can use a shared data pointer, memcpy() the
  // current value to the newly allocated NAPI buffer.
  memcpy(array_buffer_data, tensor_data, byte_length);

  // This method will only return uint8 arrays.
  nstatus = napi_create_typedarray(env, napi_uint8_array, byte_length,
                                   array_buffer_value, 0, result);
  ENSURE_NAPI_OK(env, nstatus);
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
  bool is_resource = false;
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
    case TF_RESOURCE:
      // We currently represent a resource handle as an `Uint8Array`.
      typed_array_type = napi_uint8_array;
      is_resource = true;
      break;
    default:
      REPORT_UNKNOWN_TF_DATA_TYPE(env,
                                  TFE_TensorHandleDataType(tfe_tensor_handle));
      return;
  }

  if (is_string) {
    CopyTFE_TensorHandleDataToStringArray(env, tfe_context, tfe_tensor_handle,
                                          result);
  } else if (is_resource) {
    CopyTFE_TensorHandleDataToResourceArray(env, tfe_context, tfe_tensor_handle,
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

  // OpAttr will be used beyond the scope of this function call. Stash ops in
  // a set for re-use instead of dynamically reallocating strings for
  // operations.
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

TFJSBackend::TFJSBackend(napi_env env)
    : next_tensor_id_(0), next_savedmodel_id_(0) {
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

  // TODO(kreeger): Add better support for this in the future through the JS
  // API. https://github.com/tensorflow/tfjs/issues/320
  std::string cpu_device_name;
  const int num_devices = TF_DeviceListCount(device_list);
  for (int i = 0; i < num_devices; i++) {
    const char *device_type =
        TF_DeviceListType(device_list, i, tf_status.status);
    ENSURE_TF_OK(env, tf_status);

    // Keep a reference to the host CPU device:
    if (strcmp(device_type, "CPU") == 0) {
      cpu_device_name =
          std::string(TF_DeviceListName(device_list, i, tf_status.status));
      ENSURE_TF_OK(env, tf_status);
    } else if (strcmp(device_type, "GPU") == 0) {
      device_name =
          std::string(TF_DeviceListName(device_list, i, tf_status.status));
      ENSURE_TF_OK(env, tf_status);
    }
  }

  // If no GPU devices found, fallback to host CPU:
  if (device_name.empty()) {
    device_name = cpu_device_name;
    is_gpu_device = false;
  } else {
    is_gpu_device = true;
  }
  TF_DeleteDeviceList(device_list);
}

TFJSBackend::~TFJSBackend() {
  for (auto &kv : tfe_handle_map_) {
    TFE_DeleteTensorHandle(kv.second);
  }
  for (auto &kv : tf_savedmodel_map_) {
    TF_AutoStatus tf_status;
    TF_DeleteSession(kv.second.first, tf_status.status);
    TF_DeleteGraph(kv.second.second);
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

int32_t TFJSBackend::InsertSavedModel(TF_Session *tf_session,
                                      TF_Graph *tf_graph) {
  // Both TF_Session and TF_Graph are required when executing SavedModel.
  // TF_Graph is used to find input/output operation from string name.
  return tf_savedmodel_map_
      .insert(std::make_pair(next_savedmodel_id_++,
                             std::make_pair(tf_session, tf_graph)))
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

  // Copy non-int32 and non-string tensors to a device. Most GPU kernels expect
  // to have int32 tensors in host memory.
  if (dtype_int32 != TF_INT32 && dtype_int32 != TF_STRING) {
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

  // Push `nullptr` to get a valid pointer in the call to `TFE_Execute()`
  // below.
  std::vector<TFE_TensorHandle *> result_handles(num_outputs, nullptr);

  int size = result_handles.size();
  TFE_Execute(tfe_op.op, result_handles.data(), &size, tf_status.status);
  ENSURE_TF_OK_RETVAL(env, tf_status, nullptr);

  napi_value output_tensor_infos;
  nstatus = napi_create_array_with_length(env, size, &output_tensor_infos);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  for (int32_t i = 0; i < num_outputs; i++) {
    TFE_TensorHandle *handle = result_handles[i];
    napi_value tensor_info_value = GenerateOutputTensorInfo(env, handle);
    // Push into output array
    nstatus = napi_set_element(env, output_tensor_infos, i, tensor_info_value);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
  }

  return output_tensor_infos;
}

/* Helper function to generate TensorInfo(used for JavaScript) from
 * TFE_TensorHandle. This helper function is used by ExecuteOp() and
 * RunSavedModel().
 */
napi_value TFJSBackend::GenerateOutputTensorInfo(napi_env env,
                                                 TFE_TensorHandle *handle) {
  napi_status nstatus;

  // Output tensor info object:
  napi_value tensor_info_value;
  nstatus = napi_create_object(env, &tensor_info_value);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

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

  return tensor_info_value;
}

napi_value TFJSBackend::LoadSavedModel(napi_env env,
                                       napi_value export_dir_value,
                                       napi_value tags_value) {
  TF_SessionOptions *session_options = TF_NewSessionOptions();

  TF_Buffer *run_options = TF_NewBufferFromString("", 0);

  std::string export_dir_string;
  napi_status nstatus;
  nstatus = GetStringParam(env, export_dir_value, export_dir_string);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
  const char *export_dir = export_dir_string.c_str();

  std::string tags;
  nstatus = GetStringParam(env, tags_value, tags);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  std::vector<const char *> tags_ptrs = splitStringByComma(tags);

  TF_Graph *graph = TF_NewGraph();

  TF_Buffer *metagraph = TF_NewBuffer();

  TF_AutoStatus tf_status;

  TF_Session *session = TF_LoadSessionFromSavedModel(
      session_options, run_options, export_dir, tags_ptrs.data(),
      tags_ptrs.size(), graph, metagraph, tf_status.status);
  // Delete objects that are necessary when loading the SavedModel but not gonna
  // be used later.
  TF_DeleteSessionOptions(session_options);
  TF_DeleteBuffer(run_options);
  TF_DeleteBuffer(metagraph);

  if (TF_GetCode(tf_status.status) != TF_OK) {
    NAPI_THROW_ERROR(env, "Failed to load SavedModel: %s",
                     TF_Message(tf_status.status));
    return nullptr;
  }

  napi_value output_session_id;
  nstatus = napi_create_int32(env, InsertSavedModel(session, graph),
                              &output_session_id);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
  return output_session_id;
}

void TFJSBackend::DeleteSavedModel(napi_env env,
                                   napi_value savedmodel_id_value) {
  int32_t savedmodel_id;
  ENSURE_NAPI_OK(
      env, napi_get_value_int32(env, savedmodel_id_value, &savedmodel_id));

  auto savedmodel_entry = tf_savedmodel_map_.find(savedmodel_id);
  if (savedmodel_entry == tf_savedmodel_map_.end()) {
    NAPI_THROW_ERROR(
        env, "Delete called on a SavedModel not found (savedmodel_id: %d)",
        savedmodel_id);
    return;
  }

  TF_AutoStatus tf_status;
  TF_DeleteSession(savedmodel_entry->second.first, tf_status.status);
  if (TF_GetCode(tf_status.status) != TF_OK) {
    NAPI_THROW_ERROR(env, "Failed to delete SavedModel: %s",
                     TF_Message(tf_status.status));
    return;
  }
  // TODO(kangyizhang): Add tests to validate TF_Session and TF_Graph are
  // deleted.
  TF_DeleteGraph(savedmodel_entry->second.second);
  tf_savedmodel_map_.erase(savedmodel_entry);
}

napi_value TFJSBackend::RunSavedModel(napi_env env,
                                      napi_value savedmodel_id_value,
                                      napi_value input_tensor_ids,
                                      napi_value input_op_names_value,
                                      napi_value output_op_names_value) {
  napi_status nstatus;
  TF_AutoStatus tf_status;

  int32_t savedmodel_id;
  nstatus = napi_get_value_int32(env, savedmodel_id_value, &savedmodel_id);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  // Get corresponding SavedModel session and graph.
  auto savedmodel_entry = tf_savedmodel_map_.find(savedmodel_id);
  if (savedmodel_entry == tf_savedmodel_map_.end()) {
    NAPI_THROW_ERROR(env, "SavedModel ID not found (savedmodel_id: %d)",
                     savedmodel_id);
    return nullptr;
  }

  std::string input_op_names;
  nstatus = GetStringParam(env, input_op_names_value, input_op_names);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
  std::string output_op_names;
  nstatus = GetStringParam(env, output_op_names_value, output_op_names);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  // Get input/output op names as vector
  std::vector<const char *> input_op_name_array =
      splitStringByComma(input_op_names);
  std::vector<const char *> output_op_name_array =
      splitStringByComma(output_op_names);

  std::vector<TF_Output> inputs;
  std::vector<TF_Output> outputs;

  uint32_t num_input_ids;
  nstatus = napi_get_array_length(env, input_tensor_ids, &num_input_ids);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  if (input_op_name_array.size() != num_input_ids) {
    NAPI_THROW_ERROR(env,
                     "Length of input op names (%d) does not match the length "
                     "of input tensors (%d).",
                     input_op_name_array.size(), num_input_ids);
    return nullptr;
  }

  std::vector<TF_Tensor *> input_values;

  for (uint32_t i = 0; i < num_input_ids; i++) {
    napi_value cur_input_id;
    nstatus = napi_get_element(env, input_tensor_ids, i, &cur_input_id);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    int32_t cur_input_tensor_id;
    nstatus = napi_get_value_int32(env, cur_input_id, &cur_input_tensor_id);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    // Find input tensor based on tensor id.
    auto tensor_entry = tfe_handle_map_.find(cur_input_tensor_id);
    if (tensor_entry == tfe_handle_map_.end()) {
      NAPI_THROW_ERROR(env, "Input Tensor ID not found (tensor_id: %d)",
                       cur_input_tensor_id);
      return nullptr;
    }
    TF_Tensor *inputTensor =
        TFE_TensorHandleResolve(tensor_entry->second, tf_status.status);

    if (TF_GetCode(tf_status.status) != TF_OK) {
      NAPI_THROW_ERROR(
          env, "Failed to get input tensor (tensor_id: %d) for session.",
          cur_input_tensor_id);
      return nullptr;
    }

    // Add input tensor into input values list.
    input_values.push_back(inputTensor);

    // The item in input_op_name_array is something like "serving_default_x:0".
    // Parse it into input op name and index for provided tensor.
    std::string name(input_op_name_array[i]);
    int index = name.find(":");
    std::string input_op_name = name.substr(0, index);
    const char *input_op_index = name.substr(index + 1).c_str();
    int input_tensor_index;
    if (strlen(input_op_index) == 0) {
      input_tensor_index = 0;
    } else {
      input_tensor_index = atoi(input_op_index);
    }

    // Add input op into input ops list.
    // TODO(kangyizhang): Store these TF_Operations somewhere so they don't need
    // to be generated  every time.
    TF_Operation *input_op = TF_GraphOperationByName(
        savedmodel_entry->second.second, input_op_name.c_str());
    if (input_op == nullptr) {
      NAPI_THROW_ERROR(env, "Input op name can not be found in the graph.");
      return nullptr;
    }
    TF_Output in = {input_op, input_tensor_index};
    inputs.push_back(in);
  }

  // Add output op into output ops list.
  for (uint32_t i = 0; i < output_op_name_array.size(); i++) {
    // The item in output_op_name_array is something like
    // "StatefulPartitionedCall:0". Parse it into output op name and index.
    std::string name(output_op_name_array[i]);
    int index = name.find(":");
    std::string output_op_name = name.substr(0, index);
    const char *output_op_index = name.substr(index + 1).c_str();
    int output_tensor_index;
    if (strlen(output_op_index) == 0) {
      output_tensor_index = 0;
    } else {
      output_tensor_index = atoi(output_op_index);
    }

    TF_Operation *output_op = TF_GraphOperationByName(
        savedmodel_entry->second.second, output_op_name.c_str());
    if (output_op == nullptr) {
      NAPI_THROW_ERROR(env, "Output op name can not be found in the graph.");
      return nullptr;
    }
    TF_Output out = {output_op, output_tensor_index};
    outputs.push_back(out);
  }

  std::vector<TF_Tensor *> output_values(outputs.size(), nullptr);

  TF_SessionRun(savedmodel_entry->second.first, nullptr, inputs.data(),
                input_values.data(), num_input_ids, outputs.data(),
                output_values.data(), output_op_name_array.size(), nullptr, 0,
                nullptr, tf_status.status);

  if (TF_GetCode(tf_status.status) != TF_OK) {
    NAPI_THROW_ERROR(env, "Session fail to run with error: %s",
                     TF_Message(tf_status.status));
    return nullptr;
  }

  napi_value output_tensor_infos;
  nstatus = napi_create_array_with_length(env, 1, &output_tensor_infos);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  // Generate output tensors for JS.
  for (uint32_t i = 0; i < output_op_name_array.size(); i++) {
    TFE_TensorHandle *tfe_handle =
        TFE_NewTensorHandle(output_values[i], tf_status.status);
    // Deallocate output TF_Tensor in C++.
    TF_DeleteTensor(output_values[i]);

    napi_value tensor_info_value = GenerateOutputTensorInfo(env, tfe_handle);
    // Push into output array
    nstatus = napi_set_element(env, output_tensor_infos, i, tensor_info_value);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
  }

  for (uint32_t i = 0; i < num_input_ids; i++) {
    // Deallocate input TF_Tensor in C++.
    TF_DeleteTensor(input_values[i]);
  }

  return output_tensor_infos;
}

napi_value TFJSBackend::GetNumOfSavedModels(napi_env env) {
  napi_status nstatus;
  napi_value num_saved_models;
  nstatus =
      napi_create_int32(env, tf_savedmodel_map_.size(), &num_saved_models);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
  return num_saved_models;
}

}  // namespace tfnodejs
