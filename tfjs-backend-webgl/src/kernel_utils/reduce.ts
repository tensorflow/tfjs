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

import {backend_util, DataType, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {ReduceProgram} from '../reduce_gpu';

type ReduceTypes = 'all'|'any'|'max'|'min'|'sum'|'prod';

export function reduce(
    x: TensorInfo, dtype: DataType, reductionType: ReduceTypes,
    backend: MathBackendWebGL): TensorInfo {
  const [batchSize, inSize] = x.shape;
  const windowSize = backend_util.computeOptimalWindowSize(inSize);
  const reduceInfo = {windowSize, inSize, batchSize};
  const program = new ReduceProgram(reduceInfo, reductionType);
  const output = backend.runWebGLProgram(program, [x], dtype);

  if (output.shape[1] === 1) {
    return output;
  }

  return reduce(output, dtype, reductionType, backend);
}
