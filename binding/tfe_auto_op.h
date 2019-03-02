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

#ifndef TF_NODEJS_TFE_AUTO_OP_H_
#define TF_NODEJS_TFE_AUTO_OP_H_

#include "tensorflow/c/eager/c_api.h"

namespace tfnodejs {

// Automatically cleans up a TF_Op instance.
class TFE_AutoOp {
 public:
  TFE_AutoOp(TFE_Op* op) : op(op) {}
  virtual ~TFE_AutoOp() { TFE_DeleteOp(op); }

  TFE_Op* op;
};

}  // namespace tfnodejs

#endif  // TF_NODEJS_TFE_AUTO_OP_H_
