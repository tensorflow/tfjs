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
import {MeanProgram} from '../mean_gpu';
import {ReduceProgram} from '../reduce_gpu';

type ReduceTypes = 'all'|'any'|'max'|'min'|'sum'|'prod'|'mean';

// Returns an array of configuration objects that describe each stage of the
// reduction.
function getReductionStages(inShape: number[]):
    Array<{inSize: number, windowSize: number, outSize: number}> {
  const stages = [];

  while (stages.length === 0 || stages[stages.length - 1].outSize !== 1) {
    const outSize: number =
        stages.length ? stages[stages.length - 1].outSize : inShape[1];
    const windowSize = backend_util.computeOptimalWindowSize(outSize);
    stages.push({
      inSize: outSize,
      windowSize,
      outSize: Math.ceil(outSize / windowSize)
    });
  }

  return stages;
}

export function reduce(
    x: TensorInfo, dtype: DataType, reductionType: ReduceTypes,
    backend: MathBackendWebGL): TensorInfo {
  const reductionStages = getReductionStages(x.shape);

  let result = x;
  for (let i = 0; i < reductionStages.length; i++) {
    const {inSize, windowSize, outSize} = reductionStages[i];

    let program: ReduceProgram|MeanProgram;
    let previousResult: TensorInfo;
    if (reductionType === 'mean') {
      program = i === 0 ?
          new MeanProgram(
              {windowSize, inSize, batchSize: x.shape[0], outSize}, inSize) :
          new MeanProgram({windowSize, inSize, batchSize: x.shape[0], outSize});
    } else {
      program = new ReduceProgram(
          {windowSize, inSize, batchSize: x.shape[0], outSize}, reductionType);
    }

    previousResult = result;
    result = backend.runWebGLProgram(program, [result], dtype);

    if (previousResult.dataId !== x.dataId) {
      backend.disposeIntermediateTensorInfo(previousResult);
    }
  }

  return result;
}
