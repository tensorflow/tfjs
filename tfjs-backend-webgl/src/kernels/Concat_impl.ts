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

import {backend_util, ConcatInputs, env, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {ConcatProgram} from '../concat_gpu';
import {ConcatPackedProgram} from '../concat_packed_gpu';

import {complex} from './Complex';
import {imag} from './Imag';
import {real} from './Real';
import {reshape} from './Reshape';

export function concatImpl(
    inputs: ConcatInputs, axis: number, backend: MathBackendWebGL): TensorInfo {
  const dtype = inputs[0].dtype;
  if (dtype === 'complex64') {
    const reals = inputs.map((t) => real({inputs: {input: t}, backend}));
    const imags = inputs.map((t) => imag({inputs: {input: t}, backend}));

    const realConcated = concatImpl(reals, axis, backend);
    const imagConcated = concatImpl(imags, axis, backend);

    const result =
        complex({inputs: {real: realConcated, imag: imagConcated}, backend});

    reals.forEach(r => backend.disposeIntermediateTensorInfo(r));
    imags.forEach(i => backend.disposeIntermediateTensorInfo(i));
    backend.disposeIntermediateTensorInfo(realConcated);
    backend.disposeIntermediateTensorInfo(imagConcated);

    return result;
  }

  if (inputs.length > env().getNumber('WEBGL_MAX_TEXTURES_IN_SHADER')) {
    const midIndex = Math.floor(inputs.length / 2);
    const leftSide = concatImpl(inputs.slice(0, midIndex), axis, backend);
    const rightSide = concatImpl(inputs.slice(midIndex), axis, backend);

    const result = concatImpl([leftSide, rightSide], axis, backend);

    backend.disposeIntermediateTensorInfo(leftSide);
    backend.disposeIntermediateTensorInfo(rightSide);

    return result;
  }

  if (env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') &&
      inputs[0].shape.length > 1) {
    const program = new ConcatPackedProgram(inputs.map(t => t.shape), axis);
    return backend.runWebGLProgram(program, inputs, dtype);
  }

  // Any concat of n-dimensional tensors across any axis can be reduced to
  // a concatenation of two-dimensional tensors across the axis 1 by first
  // partitioning the axes of the original tensors into those less than the
  // axis to be concatenated and the rest. Then reshape the tensors
  // into a two-dimensional tensor by collapsing these two sets of axes and
  // concatenate the resulting matrices across the axis 1, finally reshaping
  // the result to have the proper shape.
  const outShape = backend_util.computeOutShape(inputs.map(t => t.shape), axis);
  const tensors2D = inputs.map(
      x => reshape({
        inputs: {x},
        attrs: {shape: [-1, util.sizeFromShape(x.shape.slice(axis))]},
        backend
      }));
  const program =
      new ConcatProgram(tensors2D.map(t => t.shape as [number, number]));
  const result = backend.runWebGLProgram(program, tensors2D, dtype);

  tensors2D.forEach(r => backend.disposeIntermediateTensorInfo(r));
  const reshapedResult =
      reshape({inputs: {x: result}, attrs: {shape: outShape}, backend});
  backend.disposeIntermediateTensorInfo(result);

  return reshapedResult;
}
