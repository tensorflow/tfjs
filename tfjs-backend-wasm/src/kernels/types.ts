/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

// This enum must align with the enum defined in cc/backend.h.
export enum CppDType {
  float32 = 0,
  int32 = 1,
  bool = 2,
  string = 3,
  complex64 = 4
}

// Must match enum in cc/fusable_activations.h.
export enum FusableActivation {
  linear = 0,
  relu = 1,
  relu6 = 2,
  prelu = 3,
  leakyrelu = 4,
  sigmoid = 5,
  elu = 6
}
