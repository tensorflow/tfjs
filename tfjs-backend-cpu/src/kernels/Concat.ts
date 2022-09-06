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

import {backend_util, Concat, ConcatAttrs, ConcatInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {complex} from './Complex';
import {concatImpl} from './Concat_impl';
import {identity} from './Identity';
import {imag} from './Imag';
import {real} from './Real';
import {reshape} from './Reshape';

export function concat(
    args: {inputs: ConcatInputs, backend: MathBackendCPU, attrs: ConcatAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {axis} = attrs;

  const $axis = util.parseAxisParam(axis, inputs[0].shape)[0];
  let outShape = backend_util.computeOutShape(inputs.map(t => t.shape), $axis);

  if (util.sizeFromShape(outShape) === 0) {
    return backend.makeTensorInfo(outShape, inputs[0].dtype, []);
  }

  // Keep only non-empty tensors (ignore tensors with 0 in their shape).
  const $inputs = inputs.filter(t => util.sizeFromShape(t.shape) > 0);
  if ($inputs.length === 1) {
    return identity({inputs: {x: $inputs[0]}, backend});
  }

  const shapes = $inputs.map(t => t.shape);
  backend_util.assertParamsConsistent(shapes, $axis);

  if ($inputs[0].dtype === 'complex64') {
    const reals = $inputs.map((t) => real({inputs: {input: t}, backend}));
    const imags = $inputs.map((t) => imag({inputs: {input: t}, backend}));

    const realConcated = concat({inputs: reals, backend, attrs: {axis: $axis}});
    const imagConcated = concat({inputs: imags, backend, attrs: {axis: $axis}});

    const result =
        complex({inputs: {real: realConcated, imag: imagConcated}, backend});

    reals.forEach(r => backend.disposeIntermediateTensorInfo(r));
    imags.forEach(i => backend.disposeIntermediateTensorInfo(i));
    backend.disposeIntermediateTensorInfo(realConcated);
    backend.disposeIntermediateTensorInfo(imagConcated);

    return result;
  }

  // Any concat of n-dimensional tensors across any axis can be reduced to
  // a concatenation of two-dimensional tensors across the axis 1 by first
  // partitioning the axes of the original tensors into those less than the
  // axis to be concatenated and the rest. Then reshape the tensors
  // into a two-dimensional tensor by collapsing these two sets of axes and
  // concatenate the resulting matrices across the axis 1, finally reshaping
  // the result to have the proper shape.
  const inputs2D = $inputs.map(t => {
    const innerSize = util.sizeFromShape(t.shape.slice($axis));
    const shape = [-1, innerSize];
    return reshape({inputs: {x: t}, backend, attrs: {shape}});
  });

  const inputsValShapes = inputs2D.map(t => {
    return {vals: backend.data.get(t.dataId).values, shape: t.shape};
  });

  // Concats 2d tensors along axis=1.
  outShape =
      backend_util.computeOutShape(inputs2D.map(t => t.shape), 1 /* axis */);
  const simplyConcat = inputs2D[0].shape[0] === 1;
  const outVals =
      concatImpl(inputsValShapes, outShape, inputs[0].dtype, simplyConcat);

  const finalOutShape =
      backend_util.computeOutShape($inputs.map(t => t.shape), $axis);

  const outInfo =
      backend.makeTensorInfo(finalOutShape, inputs[0].dtype, outVals);

  inputs2D.forEach(t => backend.disposeIntermediateTensorInfo(t));

  return outInfo;
}

export const concatConfig: KernelConfig = {
  kernelName: Concat,
  backendName: 'cpu',
  kernelFunc: concat as {} as KernelFunc
};
