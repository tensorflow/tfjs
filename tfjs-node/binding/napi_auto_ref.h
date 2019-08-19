/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

#ifndef TF_NODEJS_NAPI_AUTO_REF_H_
#define TF_NODEJS_NAPI_AUTO_REF_H_

#include <node_api.h>
#include "utils.h"

namespace tfnodejs {

// Helper class to automatically cleanup napi_ref instances.
class NapiAutoRef {
 public:
  NapiAutoRef() : env_(nullptr), ref_(nullptr) {}

  napi_status Init(napi_env env, napi_value value) {
    env_ = env;
    return napi_create_reference(env_, value, 1, &ref_);
  }

  napi_status Cleanup() {
    if (!env_ || !ref_) {
#if DEBUG
      NAPI_THROW_ERROR(env_, "Uninitialized reference attempted to cleanup");
#endif
      return napi_invalid_arg;
    }

    napi_status nstatus = napi_delete_reference(env_, ref_);
    env_ = nullptr;
    ref_ = nullptr;
    return nstatus;
  }

  virtual ~NapiAutoRef() {
    if (env_) {
#if DEBUG
      NAPI_THROW_ERROR(env_, "Non-cleaned up napi_env instance");
#endif
    }
    if (ref_) {
#if DEBUG
      NAPI_THROW_ERROR(env_, "Non-cleaned up napi_ref instance");
#endif
    }
  }

 private:
  napi_env env_;
  napi_ref ref_;
};

}  // namespace tfnodejs

#endif  // TF_NODEJS_NAPI_AUTO_REF_H_
