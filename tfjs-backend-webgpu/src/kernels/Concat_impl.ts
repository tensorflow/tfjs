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
import {concatImplCPU} from '../kernel_utils/shared';

import {complex} from './Complex';
import {ConcatProgram} from '../concat_webgpu';
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

  let runOnCpu = backend.shouldExecuteOnCPU(inputs);

  // Run on cpu if dtype is string. For string, the backend represents it
  // as Uint8Array[], where each Uint8Array is a character. Given that the
  // computation is only on the outer array, uploading the whole data onto
  // gpu is wasteful. Also, currently webgpu doesn't have a design to
  // upload and retrieve Uint8Array[] between cpu and gpu. Therefore, we
  // just run the kernel on cpu if dtype is string.
  if (dtype === 'string') {
    runOnCpu = true;
  }

  if (runOnCpu) {
    // Any concat of n-dimensional tensors across any axis can be reduced to
    // a concatenation of two-dimensional tensors across the axis 1 by first
    // partitioning the axes of the original tensors into those less than the
    // axis to be concatenated and the rest. Then reshape the tensors
    // into a two-dimensional tensor by collapsing these two sets of axes and
    // concatenate the resulting matrices across the axis 1, finally reshaping
    // the result to have the proper shape.
    const tensors2D = inputs.map(t => {
      const innerSize = util.sizeFromShape(t.shape.slice(axis));
      const shape = [-1, innerSize];
      return reshape({inputs: {x: t}, backend, attrs: {shape}});
    });

    const inputsValShapes = tensors2D.map(t => {
      return {vals: backend.readSync(t.dataId), shape: t.shape};
    });

    // Concats 2d tensors along axis=1.
    const outShape =
        backend_util.computeOutShape(tensors2D.map(t => t.shape), 1 /* axis */);
    const simplyConcat = tensors2D[0].shape[0] === 1;
    const outVals =
        concatImplCPU(inputsValShapes, outShape, dtype, simplyConcat);

    const finalOutShape =
        backend_util.computeOutShape(inputs.map(t => t.shape), axis);

    const outInfo = backend.makeTensorInfo(finalOutShape, dtype, outVals);

    tensors2D.forEach(t => backend.disposeData(t.dataId));

    return outInfo;
  }

  const {tensors2D, outShape} = computeTensors2D(inputs, axis, backend);
  const shapes = (tensors2D).map(t => t.shape as [number, number]);
  const program = new ConcatProgram(shapes);

  const uniformData: Array<{type: string; data: number[]}> = [];
  const offsets: number[] = new Array(shapes.length - 1);
  if (offsets.length > 0) {
    offsets[0] = shapes[0][1];
    uniformData.push({type: 'int32', data: [offsets[0]]});
    for (let i = 1; i < offsets.length; i++) {
      offsets[i] = offsets[i - 1] + shapes[i][1];
      uniformData.push({type: 'int32', data: [offsets[i]]});
    }
  }

  const res = backend.runWebGPUProgram(
      program, tensors2D, tensors2D[0].dtype, uniformData);
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
