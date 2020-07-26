/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {DataId, DataType, KernelBackend, TensorInfo} from '@tensorflow/tfjs-core';

// Backend kernels should use this reshape instead of the reshape kernel.

// Internally, reshape does not create new `TensorData`, it will only increase
// the reference to the existing `TensorData`, and return a `TensorInfo` with
// the required shape.
export function reshapeImpl(
    dataId: DataId, shape: number[], dtype: DataType,
    backend: KernelBackend): TensorInfo {
  backend.incRef(dataId);

  return {dataId, shape, dtype};
}
