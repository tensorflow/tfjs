/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {backend_util, DataType, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {ReduceProgram} from '../kernels/reduce_webgpu';

type ReduceTypes = 'max'|'min'|'sum';

export function reduce(
    x: TensorInfo, dtype: DataType, reduceType: ReduceTypes,
    backend: WebGPUBackend): TensorInfo {
  const batchSize = x.shape[0];
  const inSize = x.shape[1];
  const windowSize = backend_util.computeOptimalWindowSize(inSize);
  const outSize = Math.ceil(inSize / windowSize);
  const reduceInfo = {windowSize, inSize, batchSize, outSize};
  const program = new ReduceProgram(reduceInfo, reduceType);
  return backend.runWebGPUProgram(program, [x], dtype);
}
