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

#include "tfe_context_env.h"
#include "../deps/tensorflow/include/tensorflow/c/eager/c_api.h"
#include "tf_auto_status.h"
#include "utils.h"

namespace tfnodejs {

static void Cleanup(napi_env env, void* data, void* hint) {
  TFEContextEnv* context_env = static_cast<TFEContextEnv*>(data);

  TF_AutoStatus tf_status;
  TFE_DeleteContext(context_env->context, tf_status.status);
  ENSURE_TF_OK(tf_status);

  delete context_env;
}

void InitAndBindTFEContextEnv(napi_env env, napi_value value) {
  TF_AutoStatus tf_status;
  TFE_ContextOptions* tfe_options = TFE_NewContextOptions();
  TFE_Context* tfe_context = TFE_NewContext(tfe_options, tf_status.status);
  ENSURE_TF_OK(tf_status);

  TFE_DeleteContextOptions(tfe_options);

  TFEContextEnv* tfe_context_env = new TFEContextEnv();
  tfe_context_env->context = tfe_context;
  tfe_context_env->env = env;

  ENSURE_NAPI_OK(
      env, napi_wrap(env, value, tfe_context_env, Cleanup, nullptr, nullptr));
}

}  // namespace tfnodejs
