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

#include "tfe_utils.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "../deps/include/tensorflow/c/eager/c_api.h"
#include "tf_auto_status.h"
#include "tf_auto_tensor.h"
#include "utils.h"

namespace tfnodejs {

// Used to hold strings beyond the lifetime of a JS call.
static std::set<std::string> ATTR_NAME_SET;

TFE_TensorHandle *CreateTFE_TensorHandleFromTypedArray(
    napi_env env, int64_t *shape, uint32_t shape_length, TF_DataType dtype,
    napi_value typed_array_value) {
  napi_status nstatus;

  napi_typedarray_type array_type;
  size_t array_length;
  void *array_data;
  nstatus =
      napi_get_typedarray_info(env, typed_array_value, &array_type,
                               &array_length, &array_data, nullptr, nullptr);
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
    NAPI_THROW_ERROR(env, "Byte size of elements differs between JS VM and TF");
    return nullptr;
  }

  // Determine the size of the buffer based on the dimensions.
  size_t num_elements = 1;
  for (size_t i = 0; i < shape_length; i++) {
    num_elements *= shape[i];
  }

  // Ensure the shape matches the length of the passed in typed-array.
  if (num_elements != array_length) {
    NAPI_THROW_ERROR(env, "Shape does not match typed-array in bindData()");
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
                                          napi_value *result) {
  napi_status nstatus;

  if (tfe_context == nullptr) {
    NAPI_THROW_ERROR(env, "Invalid TFE_Context");
    return;
  }
  if (tfe_tensor_handle == nullptr) {
    NAPI_THROW_ERROR(env, "Invalid TFE_TensorHandle");
    return;
  }

  // Determine the type of the array
  napi_typedarray_type array_type;
  switch (TFE_TensorHandleDataType(tfe_tensor_handle)) {
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
                                  TFE_TensorHandleDataType(tfe_tensor_handle));
      return;
  }

  TF_AutoStatus tf_status;

  TF_AutoTensor tensor(
      TFE_TensorHandleResolve(tfe_tensor_handle, tf_status.status));
  ENSURE_TF_OK(env, tf_status);

  // Determine the length of the array based on the shape of the tensor.
  size_t length = 0;
  uint32_t num_dims = TF_NumDims(tensor.tensor);
  if (num_dims == 0) {
    length = 1;
  } else {
    for (uint32_t i = 0; i < num_dims; i++) {
      if (i == 0) {
        length = TF_Dim(tensor.tensor, i);
      } else {
        length *= TF_Dim(tensor.tensor, i);
      }
    }
  }

  size_t byte_length = TF_TensorByteSize(tensor.tensor);

  napi_value array_buffer_value;
  void *array_buffer_data;
  nstatus = napi_create_arraybuffer(env, byte_length, &array_buffer_data,
                                    &array_buffer_value);
  ENSURE_NAPI_OK(env, nstatus);

  // TFE_TensorHandleResolve can use a shared data pointer, memcpy() the current
  // value to the newly allocated NAPI buffer.
  memcpy(array_buffer_data, TF_TensorData(tensor.tensor), byte_length);

  nstatus = napi_create_typedarray(env, array_type, length, array_buffer_value,
                                   0, result);
  ENSURE_NAPI_OK(env, nstatus);
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

void AssignOpAttr(napi_env env, TFE_Op *tfe_op, napi_value attr_value,
                  tfnodejs::TF_ScopedStrings* scoped_strings) {
  napi_status nstatus;

  napi_value attr_name_value;
  nstatus = napi_get_named_property(env, attr_value, "name", &attr_name_value);
  ENSURE_NAPI_OK(env, nstatus);

  char attr_name_string[NAPI_STRING_SIZE];
  nstatus = napi_get_value_string_utf8(env, attr_name_value, attr_name_string,
                                       NAPI_STRING_SIZE, nullptr);
  ENSURE_NAPI_OK(env, nstatus);

  // OpAttr will be used beyond the scope of this function call. Stash ops in a
  // set for re-use instead of dynamically reallocating strings for operations.
  const char *attr_name = ATTR_NAME_SET.insert(attr_name_string).first->c_str();

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
      // TODO(kreeger): Drop this class when 1.11 TensorFlow is released:
      // https://github.com/tensorflow/tfjs-node/pull/146#discussion_r210160129
      ENSURE_VALUE_IS_NOT_NULL(env, scoped_strings);
      std::string *str_value = scoped_strings->GetString(env, js_value);
      ENSURE_VALUE_IS_NOT_NULL(env, str_value);

      TFE_OpSetAttrString(tfe_op, attr_name, str_value->c_str(),
                          str_value->size());
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

}  // namespace tfnodejs
