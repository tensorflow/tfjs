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

#include "tfe_execute.h"
#include <set>
#include <string>
#include <vector>
#include "../deps/tensorflow/include/tensorflow/c/c_api.h"
#include "../deps/tensorflow/include/tensorflow/c/eager/c_api.h"
#include "tensor_handle.h"
#include "tf_auto_status.h"
#include "tfe_auto_op.h"
#include "tfe_context_env.h"
#include "utils.h"

namespace tfnodejs {

// Used to hold strings beyond the lifetime of a JS call.
std::set<std::string> ATTR_NAME_SET;

void AssignOpAttr(napi_env env, TFE_Op* tfe_op, napi_value attr_value) {
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
  const char* attr_name;
  auto result = ATTR_NAME_SET.find(attr_name_string);
  if (result == ATTR_NAME_SET.end()) {
    auto insert_result = ATTR_NAME_SET.insert(std::string(attr_name_string));
    // TODO assert success?
    result = insert_result.first;
  }
  attr_name = (*result).c_str();

  napi_value attr_type_value;
  nstatus = napi_get_named_property(env, attr_value, "type", &attr_type_value);
  ENSURE_NAPI_OK(env, nstatus);

  TF_AttrType tf_attr_type;
  nstatus = napi_get_value_int32(env, attr_type_value,
                                 reinterpret_cast<int32_t*>(&tf_attr_type));
  ENSURE_NAPI_OK(env, nstatus);

  napi_value type_input_value;
  nstatus =
      napi_get_named_property(env, attr_value, "value", &type_input_value);
  ENSURE_NAPI_OK(env, nstatus);

  switch (tf_attr_type) {
    case TF_ATTR_STRING: {
      char value[NAPI_STRING_SIZE];
      nstatus = napi_get_value_string_utf8(env, type_input_value, value,
                                           NAPI_STRING_SIZE, nullptr);
      ENSURE_NAPI_OK(env, nstatus);

      TFE_OpSetAttrString(tfe_op, attr_name, value);
      break;
    }

    case TF_ATTR_INT: {
      int64_t value;
      nstatus = napi_get_value_int64(env, type_input_value, &value);
      ENSURE_NAPI_OK(env, nstatus);

      TFE_OpSetAttrInt(tfe_op, attr_name, value);
      break;
    }

    case TF_ATTR_BOOL: {
      bool value;
      nstatus = napi_get_value_bool(env, type_input_value, &value);
      ENSURE_NAPI_OK(env, nstatus);

      TFE_OpSetAttrBool(tfe_op, attr_name, value);
      break;
    }

    case TF_ATTR_TYPE: {
      TF_DataType tf_data_type;
      nstatus = napi_get_value_int32(env, type_input_value,
                                     reinterpret_cast<int32_t*>(&tf_data_type));
      ENSURE_NAPI_OK(env, nstatus);

      TFE_OpSetAttrType(tfe_op, attr_name, tf_data_type);
      break;
    }

    case TF_ATTR_SHAPE: {
      std::vector<int64_t> shape_vector;
      ExtractArrayShape(env, type_input_value, &shape_vector);

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

void ExecuteOp(napi_env env, napi_value context, const char* opName,
               napi_value op_attr_inputs, napi_value inputs,
               napi_value output_tensor_array) {
  napi_status nstatus;

  TFEContextEnv* context_env;
  nstatus = napi_unwrap(env, context, reinterpret_cast<void**>(&context_env));
  ENSURE_NAPI_OK(env, nstatus);

  TF_AutoStatus tf_status;
  TFE_AutoOp tfe_op(TFE_NewOp(context_env->context, opName, tf_status.status));
  ENSURE_TF_OK(env, tf_status);

  // Assign inputs
  uint32_t inputs_length;
  nstatus = napi_get_array_length(env, inputs, &inputs_length);
  ENSURE_NAPI_OK(env, nstatus);

  for (uint32_t i = 0; i < inputs_length; i++) {
    napi_value cur_input;
    nstatus = napi_get_element(env, inputs, i, &cur_input);
    ENSURE_NAPI_OK(env, nstatus);

    WrappedTensorHandle* handle;
    nstatus = napi_unwrap(env, cur_input, reinterpret_cast<void**>(&handle));
    ENSURE_NAPI_OK(env, nstatus);

    TFE_OpAddInput(tfe_op.op, handle->handle, tf_status.status);
    ENSURE_TF_OK(env, tf_status);
  }

  uint32_t op_attrs_length;
  nstatus = napi_get_array_length(env, op_attr_inputs, &op_attrs_length);
  ENSURE_NAPI_OK(env, nstatus);

  for (uint32_t i = 0; i < op_attrs_length; i++) {
    napi_value cur_op_attr;
    nstatus = napi_get_element(env, op_attr_inputs, i, &cur_op_attr);
    ENSURE_NAPI_OK(env, nstatus);

    AssignOpAttr(env, tfe_op.op, cur_op_attr);

    // Check to see if an exception exists, if so return a failure.
    bool has_exception = false;
    nstatus = napi_is_exception_pending(env, &has_exception);
    ENSURE_NAPI_OK(env, nstatus);
    if (has_exception) {
      return;
    }
  }

  // Number of outputs will match the passed in output tensor handles.
  uint32_t output_length;
  nstatus = napi_get_array_length(env, output_tensor_array, &output_length);
  ENSURE_NAPI_OK(env, nstatus);

  // Push `nullptr` to get a valid pointer in the call to `TFE_Execute()` below.
  std::vector<TFE_TensorHandle*> result_handles;
  for (uint32_t i = 0; i < output_length; i++) {
    result_handles.push_back(nullptr);
  }

  int size = result_handles.size();
  TFE_Execute(tfe_op.op, result_handles.data(), &size, tf_status.status);
  ENSURE_TF_OK(env, tf_status);

  // Swap pointer on the output tensor handles.
  for (uint32_t i = 0; i < output_length; i++) {
    napi_value output_value;
    nstatus = napi_get_element(env, output_tensor_array, i, &output_value);
    ENSURE_NAPI_OK(env, nstatus);

    WrappedTensorHandle* handle;
    nstatus = napi_unwrap(env, output_value, reinterpret_cast<void**>(&handle));
    ENSURE_NAPI_OK(env, nstatus);
    // Ensure that handle is from an unused tensor handle so no cleanup is
    // needed.
    // TODO(kreeger): If handle reuse, this needs to be tweaked.
    if (handle->handle != nullptr) {
      NAPI_THROW_ERROR(
          env, "Invalid output Tensor not built with default constructor");
      return;
    }

    handle->handle = result_handles[i];
  }
}

}  // namespace tfnodejs
