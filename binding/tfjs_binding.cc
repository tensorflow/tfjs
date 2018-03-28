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
  ENSURE_CONSTRUCTOR_CALL_RETVAL(env, info, nullptr);

  napi_status nstatus;

  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, 0, nullptr, &js_this, nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, js_this);

  InitAndBindTFEContextEnv(env, js_this);
  return js_this;
}

static napi_value NewTensorHandle(napi_env env, napi_callback_info info) {
  ENSURE_CONSTRUCTOR_CALL_RETVAL(env, info, nullptr);

  napi_status nstatus;
  size_t argc = 0;
  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, &argc, nullptr, &js_this, nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, js_this);

  InitTensorHandle(env, js_this);
  return js_this;
}

static napi_value CopyTensorHandleBuffer(napi_env env,
                                         napi_callback_info info) {
  napi_status nstatus;

  // Binding buffer takes 3 params: shape, dtype, buffer.
  size_t argc = 3;
  napi_value args[argc];
  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, &argc, args, &js_this, nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, js_this);

  if (argc < 3) {
    NAPI_THROW_ERROR(env, "Invalid number of arguments passed to bindBuffer()");
    return js_this;
  }

  // Param 0 shoud be shape:
  napi_value shape_value = args[0];
  ENSURE_VALUE_IS_ARRAY_RETVAL(env, shape_value, js_this);

  std::vector<int64_t> shape_vector;
  ExtractArrayShape(env, shape_value, &shape_vector);

  // Param 1 should be dtype:
  napi_value dtype_arg = args[1];
  int32_t dtype_int32_val;
  nstatus = napi_get_value_int32(env, dtype_arg, &dtype_int32_val);
  TF_DataType dtype = static_cast<TF_DataType>(dtype_int32_val);

  // Param 2 should be typed-array:
  napi_value typed_array_value = args[2];
  ENSURE_VALUE_IS_TYPED_ARRAY_RETVAL(env, typed_array_value, js_this);

  CopyTensorJSBuffer(env, js_this, shape_vector.data(), shape_vector.size(),
                     dtype, typed_array_value);

  return js_this;
}

static napi_value GetTensorHandleData(napi_env env, napi_callback_info info) {
  napi_status nstatus;

  size_t argc = 1;
  napi_value context_value;
  napi_value js_this;
  nstatus =
      napi_get_cb_info(env, info, &argc, &context_value, &js_this, nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, js_this);

  if (argc < 1) {
    NAPI_THROW_ERROR(env, "Invalid number of arguments passed to dataSync()");
    return js_this;
  }

  napi_value result;
  GetTensorData(env, context_value, js_this, &result);
  return result;
}

static napi_value GetTensorHandleShape(napi_env env, napi_callback_info info) {
  napi_status nstatus;

  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, 0, nullptr, &js_this, nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, js_this);

  napi_value result;
  GetTensorShape(env, js_this, &result);
  return result;
}

static napi_value GetTensorHandleDtype(napi_env env, napi_callback_info info) {
  napi_status nstatus;

  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, 0, nullptr, &js_this, nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, js_this);

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
  ENSURE_NAPI_OK_RETVAL(env, nstatus, js_this);

  // Ensure arguments:
  if (argc < 5) {
    NAPI_THROW_ERROR(env, "Invalid number of arguments passed to execute()");
    return js_this;
  }
  ENSURE_VALUE_IS_OBJECT_RETVAL(env, args[0], js_this);
  ENSURE_VALUE_IS_STRING_RETVAL(env, args[1], js_this);
  ENSURE_VALUE_IS_ARRAY_RETVAL(env, args[2], js_this);
  ENSURE_VALUE_IS_ARRAY_RETVAL(env, args[3], js_this);
  ENSURE_VALUE_IS_ARRAY_RETVAL(env, args[4], js_this);

  char op_name[NAPI_STRING_SIZE];
  nstatus = napi_get_value_string_utf8(env, args[1], op_name, NAPI_STRING_SIZE,
                                       nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, js_this);

  ExecuteOp(env,
            args[0],  // TFE_Context wrapper
            op_name,
            args[2],   // TFEOpAttr array
            args[3],   // TensorHandle array
            args[4]);  // Output TensorHandle array.
  return js_this;
}

static napi_value InitTFNodeJSBinding(napi_env env, napi_value exports) {
  napi_status nstatus;

  // TFE Context class
  napi_value context_class;
  nstatus = napi_define_class(env, "Context", NAPI_AUTO_LENGTH, NewContext,
                              nullptr, 0, nullptr, &context_class);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, exports);

  // Tensor Handle class
  napi_property_descriptor tensor_handle_properties[] = {
      {"copyBuffer", nullptr, CopyTensorHandleBuffer, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"dataSync", nullptr, GetTensorHandleData, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"shape", nullptr, nullptr, GetTensorHandleShape, nullptr, nullptr,
       napi_default, nullptr},
      {"dtype", nullptr, nullptr, GetTensorHandleDtype, nullptr, nullptr,
       napi_default, nullptr}};
  ;

  napi_value tensor_handle_class;
  nstatus =
      napi_define_class(env, "TensorHandle", NAPI_AUTO_LENGTH, NewTensorHandle,
                        nullptr, ARRAY_SIZE(tensor_handle_properties),
                        tensor_handle_properties, &tensor_handle_class);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, exports);

  // TF version
  napi_value tf_version;
  nstatus = napi_create_string_latin1(env, TF_Version(), -1, &tf_version);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, exports);

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
  ENSURE_NAPI_OK_RETVAL(env, nstatus, exports);

  // Export TF property types to JS
#define EXPORT_INT_PROPERTY(v) AssignIntProperty(env, exports, #v, v)
  // Types
  EXPORT_INT_PROPERTY(TF_FLOAT);
  EXPORT_INT_PROPERTY(TF_INT32);
  EXPORT_INT_PROPERTY(TF_BOOL);

  // Op AttrType
  EXPORT_INT_PROPERTY(TF_ATTR_STRING);
  EXPORT_INT_PROPERTY(TF_ATTR_INT);
  EXPORT_INT_PROPERTY(TF_ATTR_FLOAT);
  EXPORT_INT_PROPERTY(TF_ATTR_BOOL);
  EXPORT_INT_PROPERTY(TF_ATTR_TYPE);
  EXPORT_INT_PROPERTY(TF_ATTR_SHAPE);
#undef EXPORT_INT_PROPERTY

  return exports;
}

NAPI_MODULE(tfe_binding, InitTFNodeJSBinding)

}  // namespace tfnodejs
