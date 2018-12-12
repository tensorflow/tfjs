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
#include "tfjs_backend.h"
#include "utils.h"

namespace tfnodejs {

TFJSBackend* gBackend = nullptr;

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

static napi_value CreateTensor(napi_env env, napi_callback_info info) {
  napi_status nstatus;

  // Create tensor takes 3 params: shape, dtype, typed-array/array:
  size_t argc = 3;
  napi_value args[3];
  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, &argc, args, &js_this, nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  if (argc < 3) {
    NAPI_THROW_ERROR(env, "Invalid number of args passed to createTensor()");
    return nullptr;
  }

  ENSURE_VALUE_IS_ARRAY_RETVAL(env, args[0], nullptr);
  ENSURE_VALUE_IS_NUMBER_RETVAL(env, args[1], nullptr);

  // The third array can either be a typed array or an array:
  bool is_typed_array;
  nstatus = napi_is_typedarray(env, args[2], &is_typed_array);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);
  if (!is_typed_array) {
    ENSURE_VALUE_IS_ARRAY_RETVAL(env, args[2], nullptr);
  }

  return gBackend->CreateTensor(env, args[0], args[1], args[2]);
}

static napi_value DeleteTensor(napi_env env, napi_callback_info info) {
  napi_status nstatus;

  // Delete tensor takes 1 param: tensor ID;
  size_t argc = 1;
  napi_value args[1];
  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, &argc, args, &js_this, nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, js_this);

  if (argc < 1) {
    NAPI_THROW_ERROR(env, "Invalid number of args passed to deleteTensor()");
    return js_this;
  }

  ENSURE_VALUE_IS_NUMBER_RETVAL(env, args[0], js_this);

  gBackend->DeleteTensor(env, args[0]);
  return js_this;
}

static napi_value TensorDataSync(napi_env env, napi_callback_info info) {
  napi_status nstatus;

  // Tensor data-sync takes 1 param: tensor ID;
  size_t argc = 1;
  napi_value args[1];
  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, &argc, args, &js_this, nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, js_this);

  if (argc < 1) {
    NAPI_THROW_ERROR(env, "Invalid number of args passed to tensorDataSync()");
    return nullptr;
  }

  ENSURE_VALUE_IS_NUMBER_RETVAL(env, args[0], js_this);

  return gBackend->GetTensorData(env, args[0]);
}

static napi_value ExecuteOp(napi_env env, napi_callback_info info) {
  napi_status nstatus;

  // Create tensor takes 3 params: op-name, op-attrs, input-tensor-ids,
  // num-outputs:
  size_t argc = 4;
  napi_value args[4];
  napi_value js_this;
  nstatus = napi_get_cb_info(env, info, &argc, args, &js_this, nullptr);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nullptr);

  if (argc < 4) {
    NAPI_THROW_ERROR(env, "Invalid number of args passed to executeOp()");
    return nullptr;
  }

  ENSURE_VALUE_IS_STRING_RETVAL(env, args[0], nullptr);
  ENSURE_VALUE_IS_ARRAY_RETVAL(env, args[1], nullptr);
  ENSURE_VALUE_IS_ARRAY_RETVAL(env, args[2], nullptr);
  ENSURE_VALUE_IS_NUMBER_RETVAL(env, args[3], nullptr);

  return gBackend->ExecuteOp(env, args[0], args[1], args[2], args[3]);
}

static napi_value InitTFNodeJSBinding(napi_env env, napi_value exports) {
  napi_status nstatus;

  gBackend = TFJSBackend::Create(env);
  ENSURE_VALUE_IS_NOT_NULL_RETVAL(env, gBackend, nullptr);

  // TF version
  napi_value tf_version;
  nstatus = napi_create_string_latin1(env, TF_Version(), -1, &tf_version);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, exports);

  // Set all export values list here.
  napi_property_descriptor exports_properties[] = {
      {"createTensor", nullptr, CreateTensor, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"deleteTensor", nullptr, DeleteTensor, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"tensorDataSync", nullptr, TensorDataSync, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"executeOp", nullptr, ExecuteOp, nullptr, nullptr, nullptr, napi_default,
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
  EXPORT_INT_PROPERTY(TF_COMPLEX64);
  EXPORT_INT_PROPERTY(TF_STRING);

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
