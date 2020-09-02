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

function getReductionSizes(inShape: number[]):
    Array<{inSize: number, windowSize: number, outSize: number}> {
  const sizes = [];

  while (sizes.length === 0 || sizes[sizes.length - 1].outSize !== 1) {
    const outSize: number =
        sizes.length ? sizes[sizes.length - 1].outSize : inShape[1];
    const windowSize = backend_util.computeOptimalWindowSize(outSize);
    sizes.push({
      inSize: outSize,
      windowSize,
      outSize: Math.ceil(outSize / windowSize)
    });
  }

  return sizes;
}

export function reduce(
    x: TensorInfo, dtype: DataType, reductionType: ReduceTypes,
    backend: MathBackendWebGL): TensorInfo {
  const sizes = getReductionSizes(x.shape);

  let result = x;
  for (let i = 0; i < sizes.length; i++) {
    const {inSize, windowSize} = sizes[i];

    const program = new ReduceProgram(
        {windowSize, inSize, batchSize: x.shape[0]}, reductionType);
    const previousResult = result;
    result = backend.runWebGLProgram(program, [result], dtype);

    if (i > 0) {
      backend.disposeData(previousResult.dataId);
    }
  }

  return result;
}
