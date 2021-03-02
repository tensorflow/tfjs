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

import {backend_util, ConcatInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {complex} from './Complex';
import {ConcatProgram} from './concat_webgpu';
import {imag} from './Imag';
import {real} from './Real';
import {reshape} from './Reshape';

export function concatImpl(
    inputs: ConcatInputs, axis: number, backend: WebGPUBackend): TensorInfo {
  const dtype = inputs[0].dtype;
  if (dtype === 'complex64') {
    const reals = inputs.map((t) => real({inputs: {input: t}, backend}));
    const imags = inputs.map((t) => imag({inputs: {input: t}, backend}));

    const realConcated = concatImpl(reals, axis, backend);
    const imagConcated = concatImpl(imags, axis, backend);

    const result =
        complex({inputs: {real: realConcated, imag: imagConcated}, backend});

    reals.forEach(r => backend.disposeData(r.dataId));
    imags.forEach(i => backend.disposeData(i.dataId));
    backend.disposeData(realConcated.dataId);
    backend.disposeData(imagConcated.dataId);

    return result;
  }

  const {tensors2D, outShape} = computeTensors2D(inputs, axis, backend);
  const program =
      new ConcatProgram((tensors2D).map(t => t.shape as [number, number]));
  const res = backend.runWebGPUProgram(program, tensors2D, tensors2D[0].dtype);
  tensors2D.forEach(r => backend.disposeData(r.dataId));

  const reshapedResult =
      reshape({inputs: {x: res}, backend, attrs: {shape: outShape}});
  backend.disposeData(res.dataId);
  return reshapedResult;
}

function computeTensors2D(
    inputs: ConcatInputs, axis: number, backend: WebGPUBackend) {
  const outShape = backend_util.computeOutShape(inputs.map(t => t.shape), axis);
  const tensors2D = inputs.map(t => reshape({
                                 inputs: {x: t},
                                 backend,
                                 attrs: {
                                   shape: [
                                     util.sizeFromShape(t.shape.slice(0, axis)),
                                     util.sizeFromShape(t.shape.slice(axis))
                                   ]
                                 }
                               }));

  return {tensors2D, outShape};
}
