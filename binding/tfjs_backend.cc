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
#include <memory>
#include "tf_scoped_strings.h"
#include "tfe_auto_op.h"
#include "tfe_utils.h"
#include "utils.h"

namespace tfnodejs {

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
    TF_AutoStatus tf_status;
    TFE_DeleteContext(tfe_context_, tf_status.status);
  }
}

TFJSBackend *TFJSBackend::Create(napi_env env) { return new TFJSBackend(env); }

int32_t TFJSBackend::InsertHandle(TFE_TensorHandle *tfe_handle) {
  return tfe_handle_map_.insert(std::make_pair(next_tensor_id_++, tfe_handle))
      .first->first;
}

napi_value TFJSBackend::CreateTensor(napi_env env, napi_value shape_value,
                                     napi_value dtype_value,
                                     napi_value typed_array_value) {
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

  TFE_TensorHandle *tfe_handle = CreateTFE_TensorHandleFromTypedArray(
      env, shape_vector.data(), shape_vector.size(),
      static_cast<TF_DataType>(dtype_int32), typed_array_value);

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
    NAPI_THROW_ERROR(env, "Delete called on a Tensor not referenced");
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
    NAPI_THROW_ERROR(env, "Get data called on a Tensor not referenced");
    return nullptr;
  }

  napi_value typed_array_value;
  CopyTFE_TensorHandleDataToTypedArray(env, tfe_context_, tensor_entry->second,
                                       &typed_array_value);
  return typed_array_value;
}

napi_value TFJSBackend::ExecuteOp(napi_env env, napi_value op_name_value,
                                  napi_value op_attr_inputs,
                                  napi_value input_tensor_ids,
                                  napi_value num_output_values) {
  napi_status nstatus;

  char op_name[NAPI_STRING_SIZE];
  nstatus = napi_get_value_string_utf8(env, op_name_value, op_name,
                                       NAPI_STRING_SIZE, nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  TF_AutoStatus tf_status;
  TFE_AutoOp tfe_op(TFE_NewOp(tfe_context_, op_name, tf_status.status));
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
      NAPI_THROW_ERROR(env, "Input Tensor ID not referenced");
      return nullptr;
    }

    TFE_OpAddInput(tfe_op.op, input_tensor_entry->second, tf_status.status);
    ENSURE_TF_OK_RETVAL(env, tf_status, nullptr);
  }

  uint32_t op_attrs_length;
  nstatus = napi_get_array_length(env, op_attr_inputs, &op_attrs_length);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  // Some Ops require heap-based string attributes, manage those scoped here:
  // TODO(kreeger): Drop this class when 1.11 TensorFlow is released:
  // https://github.com/tensorflow/tfjs-node/pull/146#discussion_r210160129
  TF_ScopedStrings scoped_strings;

  for (uint32_t i = 0; i < op_attrs_length; i++) {
    napi_value cur_op_attr;
    nstatus = napi_get_element(env, op_attr_inputs, i, &cur_op_attr);
    ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

    AssignOpAttr(env, tfe_op.op, cur_op_attr, &scoped_strings);

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
