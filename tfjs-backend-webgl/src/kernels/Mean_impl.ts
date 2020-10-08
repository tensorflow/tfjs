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

import {backend_util, DataType, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {reshape} from '../kernels/Reshape';
import {MeanProgram} from '../mean_gpu';

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

function reduce(
    x: TensorInfo, dtype: DataType, backend: MathBackendWebGL): TensorInfo {
  const reductionStages = getReductionStages(x.shape);

  let result = x;
  for (let i = 0; i < reductionStages.length; i++) {
    const {inSize, windowSize, outSize} = reductionStages[i];

    const program =
        new MeanProgram({windowSize, inSize, batchSize: x.shape[0], outSize});
    const previousResult = result;
    result = backend.runWebGLProgram(program, [result], dtype);

    if (previousResult.dataId !== x.dataId) {
      backend.disposeData(previousResult.dataId);
    }
  }

  return result;
}

export function meanImpl(
    x: TensorInfo, reduceShape: number[], outShape: number[],
    backend: MathBackendWebGL): TensorInfo {
  const inSize = util.sizeFromShape(reduceShape);
  const xSize = util.sizeFromShape(x.shape);
  const batchSize = xSize / inSize;
  const reshapedInput =
      reshape({inputs: {x}, attrs: {shape: [batchSize, inSize]}, backend});

  const reduced = reduce(reshapedInput, x.dtype, backend);
  const reshapedOutput =
      reshape({inputs: {x: reduced}, attrs: {shape: outShape}, backend});

  backend.disposeIntermediateTensorInfo(reshapedInput);
  backend.disposeIntermediateTensorInfo(reduced);

  return reshapedOutput;
}
