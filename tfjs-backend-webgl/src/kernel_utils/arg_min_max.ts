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

import {backend_util, env, TensorInfo, util} from '@tensorflow/tfjs-core';

import {ArgMinMaxProgram} from '../argminmax_gpu';
import {ArgMinMaxPackedProgram} from '../argminmax_packed_gpu';
import {MathBackendWebGL} from '../backend_webgl';
import {reshape} from '../kernels/Reshape';

function argReduce(
    backend: MathBackendWebGL, x: TensorInfo, reduceType: 'max'|'min',
    bestIndicesA: TensorInfo = null): TensorInfo {
  let batchSize = x.shape[0];
  let inSize = x.shape[1];
  if (bestIndicesA != null) {
    batchSize = bestIndicesA.shape[0];
    inSize = bestIndicesA.shape[1];
  }
  const windowSize = backend_util.computeOptimalWindowSize(inSize);
  const reduceInfo =
      {windowSize, inSize, batchSize, outSize: Math.ceil(inSize / windowSize)};
  const program =
      new ArgMinMaxProgram(reduceInfo, reduceType, bestIndicesA == null);
  const inputs = [x];
  if (bestIndicesA != null) {
    inputs.push(bestIndicesA);
  }
  const output = backend.runWebGLProgram(program, inputs, 'int32');
  // No need to run another GPGPU program.
  if (output.shape[1] === 1) {
    return output;
  }
  const result = argReduce(backend, x, reduceType, output);
  backend.disposeIntermediateTensorInfo(output);
  return result;
}

function argReducePacked(
    backend: MathBackendWebGL, x: TensorInfo, reduceType: 'max'|'min',
    bestIndicesA: TensorInfo = null): TensorInfo {
  const inShape = bestIndicesA != null ? bestIndicesA.shape : x.shape;
  const inSize = inShape[inShape.length - 1];
  const windowSize = backend_util.computeOptimalWindowSize(inSize);
  const program = new ArgMinMaxPackedProgram(
      inShape, windowSize, reduceType, bestIndicesA == null);
  const inputs = bestIndicesA == null ? [x] : [x, bestIndicesA];
  const output = backend.runWebGLProgram(program, inputs, 'int32');
  if (output.shape.length === x.shape.length) {
    const result = argReducePacked(backend, x, reduceType, output);
    backend.disposeIntermediateTensorInfo(output);
    return result;
  }
  return output;
}

export function argMinMaxReduce(
    backend: MathBackendWebGL, x: TensorInfo, axis: number,
    reduceType: 'min'|'max'): TensorInfo {
  const axes = [axis];
  backend_util.assertAxesAreInnerMostDims(
      'arg' + reduceType.charAt(0).toUpperCase() + reduceType.slice(1), axes,
      x.shape.length);
  if (!env().getBool('WEBGL_PACK_REDUCE') || x.shape.length <= 2) {
    const intermediateTensorInfos = [];
    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = reshape({inputs: {x}, backend, attrs: {shape: [-1, inSize]}});
    intermediateTensorInfos.push(a2D);

    const reduced = argReduce(backend, a2D, reduceType);
    intermediateTensorInfos.push(reduced);
    const reshaped =
        reshape({inputs: {x: reduced}, backend, attrs: {shape: outShape}});

    intermediateTensorInfos.forEach(
        t => backend.disposeIntermediateTensorInfo(t));
    return reshaped;
  }
  return argReducePacked(backend, x, reduceType);
}
