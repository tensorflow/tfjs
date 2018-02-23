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

#include <node_api.h>
#include "tensor_handle.h"
#include "tfe_context_env.h"
#include "tfe_execute.h"
#include "utils.h"

namespace tfnodejs {

static void AssignIntProperty(napi_env env, napi_value exports,
                              const char* name, int32_t value) {
  napi_value js_value;
  napi_status nstatus = napi_create_int32(env, value, &js_value);
  ENSURE_NAPI_OK(env, nstatus);

  napi_property_descriptor property = {name,         nullptr, nullptr,
                                       nullptr,      nullptr, js_value,
                                       napi_default, nullptr};
  nstatus = napi_define_properties(env, exports, 1, &property);
  ENSURE_NAPI_OK(env, nstatus);
}

static napi_value NewContext(napi_env env, napi_callback_info info) {
  ENSURE_CONSTRUCTOR_CALL(env, info);

  napi_status nstatus;

  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, 0, nullptr, &js_this, nullptr);
  ENSURE_NAPI_OK(env, nstatus);

  InitAndBindTFEContextEnv(env, js_this);
  return js_this;
}

static napi_value NewTensorHandle(napi_env env, napi_callback_info info) {
  ENSURE_CONSTRUCTOR_CALL(env, info);

  napi_status nstatus;

  // This method takes two arguments - shape and dtype.
  size_t argc = 2;
  napi_value args[argc];
  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, &argc, args, &js_this, nullptr);
  ENSURE_NAPI_OK(env, nstatus);

  napi_value shape_value = args[0];
  ENSURE_VALUE_IS_ARRAY(env, shape_value);

  uint32_t shape_length;
  nstatus = napi_get_array_length(env, shape_value, &shape_length);
  ENSURE_VALUE_IS_LESS_THAN(shape_length, MAX_TENSOR_SHAPE);

  int64_t shape[shape_length];
  for (uint32_t i = 0; i < shape_length; i++) {
    napi_value dimension_value;
    nstatus = napi_get_element(env, shape_value, i, &dimension_value);
    ENSURE_NAPI_OK(env, nstatus);

    nstatus = napi_get_value_int64(env, dimension_value, &shape[i]);
    ENSURE_NAPI_OK(env, nstatus);
  }

  napi_value dtype_arg = args[1];
  int32_t dtype_int32_val;
  nstatus = napi_get_value_int32(env, dtype_arg, &dtype_int32_val);
  TF_DataType dtype = static_cast<TF_DataType>(dtype_int32_val);

  InitTensorHandle(env, js_this, shape, shape_length, dtype);
  return js_this;
}

static napi_value SetTensorHandleBuffer(napi_env env, napi_callback_info info) {
  napi_status nstatus;

  // This method should take only one param - a typed array.
  size_t argc = 1;
  napi_value typed_array_value;
  napi_value js_this;
  nstatus =
      napi_get_cb_info(env, info, &argc, &typed_array_value, &js_this, nullptr);
  ENSURE_NAPI_OK(env, nstatus);

  ENSURE_VALUE_IS_TYPED_ARRAY(env, typed_array_value);
  BindTensorJSBuffer(env, js_this, typed_array_value);

  return js_this;
}

static napi_value GetTensorHandleData(napi_env env, napi_callback_info info) {
  napi_status nstatus;

  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, 0, nullptr, &js_this, nullptr);
  ENSURE_NAPI_OK(env, nstatus);

  napi_value result;
  GetTensorData(env, js_this, &result);
  return result;
}

static napi_value GetTensorHandleShape(napi_env env, napi_callback_info info) {
  napi_status nstatus;

  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, 0, nullptr, &js_this, nullptr);
  ENSURE_NAPI_OK(env, nstatus);

  napi_value result;
  GetTensorShape(env, js_this, &result);
  return result;
}

static napi_value GetTensorHandleDtype(napi_env env, napi_callback_info info) {
  napi_status nstatus;

  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, 0, nullptr, &js_this, nullptr);
  ENSURE_NAPI_OK(env, nstatus);

  napi_value result;
  GetTensorDtype(env, js_this, &result);
  return result;
}

static napi_value ExecuteTFE(napi_env env, napi_callback_info info) {
  napi_status nstatus;

  size_t argc = 5;
  napi_value args[argc];
  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, &argc, args, &js_this, nullptr);
  ENSURE_NAPI_OK(env, nstatus);

  // TODO - assert that the proper number of values is passed in.

  char op_name[NAPI_STRING_SIZE];
  nstatus = napi_get_value_string_utf8(env, args[1], op_name, NAPI_STRING_SIZE,
                                       nullptr);
  ENSURE_NAPI_OK(env, nstatus);

  ExecuteOp(env,
            args[0],  // TFE_Context wrapper
            op_name,
            args[2],   // TFEOpAttr array
            args[3],   // TensorHandle array
            args[4]);  // Output TensorHandle.
  return js_this;
}

static napi_value InitTFNodeJSBinding(napi_env env, napi_value exports) {
  napi_status nstatus;

  // TFE Context class
  napi_value context_class;
  nstatus = napi_define_class(env, "Context", NAPI_AUTO_LENGTH, NewContext,
                              nullptr, 0, nullptr, &context_class);
  ENSURE_NAPI_OK(env, nstatus);

  // Tensor Handle class
  napi_property_descriptor tensor_handle_properties[] = {
      {"bindBuffer", nullptr, SetTensorHandleBuffer, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"data", nullptr, GetTensorHandleData, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"shape", nullptr, nullptr, GetTensorHandleShape, nullptr, nullptr,
       napi_default, nullptr},
      {"dtype", nullptr, nullptr, GetTensorHandleDtype, nullptr, nullptr,
       napi_default, nullptr}};

  napi_value tensor_handle_class;
  nstatus =
      napi_define_class(env, "TensorHandle", NAPI_AUTO_LENGTH, NewTensorHandle,
                        nullptr, ARRAY_SIZE(tensor_handle_properties),
                        tensor_handle_properties, &tensor_handle_class);
  ENSURE_NAPI_OK(env, nstatus);

  // TF version
  napi_value tf_version;
  nstatus = napi_create_string_latin1(env, TF_Version(), -1, &tf_version);
  ENSURE_NAPI_OK(env, nstatus);

  // Set all export values list here.
  napi_property_descriptor exports_properties[] = {
      {"Context", nullptr, nullptr, nullptr, nullptr, context_class,
       napi_default, nullptr},
      {"TensorHandle", nullptr, nullptr, nullptr, nullptr, tensor_handle_class,
       napi_default, nullptr},
      {"execute", nullptr, ExecuteTFE, nullptr, nullptr, nullptr, napi_default,
       nullptr},
      {"TF_Version", nullptr, nullptr, nullptr, nullptr, tf_version,
       napi_default, nullptr},
  };
  nstatus = napi_define_properties(env, exports, ARRAY_SIZE(exports_properties),
                                   exports_properties);
  ENSURE_NAPI_OK(env, nstatus);

  // Export TF property types to JS
#define EXPORT_INT_PROPERTY(v) AssignIntProperty(env, exports, #v, v)
  // Types
  EXPORT_INT_PROPERTY(TF_FLOAT);
  EXPORT_INT_PROPERTY(TF_INT32);
  EXPORT_INT_PROPERTY(TF_BOOL);

  // Op AttrType
  EXPORT_INT_PROPERTY(TF_ATTR_STRING);
  EXPORT_INT_PROPERTY(TF_ATTR_INT);
  EXPORT_INT_PROPERTY(TF_ATTR_BOOL);
  EXPORT_INT_PROPERTY(TF_ATTR_TYPE);
  EXPORT_INT_PROPERTY(TF_ATTR_SHAPE);
  EXPORT_INT_PROPERTY(TF_ATTR_TENSOR);
  EXPORT_INT_PROPERTY(TF_ATTR_PLACEHOLDER);
  EXPORT_INT_PROPERTY(TF_ATTR_FUNC);
#undef EXPORT_INT_PROPERTY

  return exports;
}

NAPI_MODULE(tfe_binding, InitTFNodeJSBinding)

}  // namespace tfnodejs
