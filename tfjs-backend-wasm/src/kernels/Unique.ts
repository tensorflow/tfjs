/**
 * @license
 * Copyright 2023 Google LLC.
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
import {KernelConfig, KernelFunc, TensorInfo, Unique, UniqueAttrs, UniqueInputs} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {uniqueImplCPU} from '../kernel_utils/shared';

function unique(
    args: {inputs: UniqueInputs, attrs: UniqueAttrs, backend: BackendWasm}):
    TensorInfo[] {
  const {inputs, attrs, backend} = args;
  const {axis} = attrs;
  const {x} = inputs;

  const {outputValues, outputShape, indices} =
      uniqueImplCPU(backend.readSync(x.dataId), axis, x.shape, x.dtype);

  return [
    backend.makeOutput(
        outputShape, x.dtype, /*memoryOffset=*/undefined, outputValues),
    backend.makeOutput(
        [indices.length], 'int32', /*memoryOffset=*/undefined, indices),
  ];
}

export const uniqueConfig: KernelConfig = {
  kernelName: Unique,
  backendName: 'wasm',
  kernelFunc: unique as unknown as KernelFunc,
};
