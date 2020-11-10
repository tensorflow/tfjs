/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

#ifndef TF_NODEJS_TF_AUTO_TENSOR_H_
#define TF_NODEJS_TF_AUTO_TENSOR_H_

#include "tensorflow/c/c_api.h"

namespace tfnodejs {

// Automatically cleans up a TF_Tensor instance.
class TF_AutoTensor {
 public:
  TF_AutoTensor(TF_Tensor* tensor) : tensor(tensor) {}
  virtual ~TF_AutoTensor() { TF_DeleteTensor(tensor); }

  TF_Tensor* tensor;
};

}  // namespace tfnodejs

#endif  // TF_NODEJS_TF_AUTO_TENSOR_H_
