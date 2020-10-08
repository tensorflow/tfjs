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

import {TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {getReductionStages} from '../kernel_utils/reduce';
import {reshape} from '../kernels/Reshape';
import {MeanProgram} from '../mean_gpu';

function meanReduce(
    x: TensorInfo, reduceSize: number, backend: MathBackendWebGL): TensorInfo {
  const reductionStages = getReductionStages(x.shape);

  let result = x;
  for (let i = 0; i < reductionStages.length; i++) {
    const {inSize, windowSize, outSize} = reductionStages[i];

    const program = i === 0 ?
        new MeanProgram(
            {windowSize, inSize, batchSize: x.shape[0], outSize}, reduceSize) :
        new MeanProgram({windowSize, inSize, batchSize: x.shape[0], outSize});
    const previousResult = result;
    result = backend.runWebGLProgram(program, [result], 'float32');

    if (previousResult.dataId !== x.dataId) {
      backend.disposeIntermediateTensorInfo(previousResult);
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

  const reduced =
      meanReduce(reshapedInput, util.sizeFromShape(reduceShape), backend);
  const reshapedOutput =
      reshape({inputs: {x: reduced}, attrs: {shape: outShape}, backend});

  backend.disposeIntermediateTensorInfo(reshapedInput);
  backend.disposeIntermediateTensorInfo(reduced);

  return reshapedOutput;
}
