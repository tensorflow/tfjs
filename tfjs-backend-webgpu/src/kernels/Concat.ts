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

import {backend_util, Concat, ConcatAttrs, ConcatInputs, KernelConfig, KernelFunc, Tensor2D, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {ConcatProgram} from './concat_webgpu';
import {reshape} from './Reshape';

export function concat(
    args:
        {inputs: ConcatInputs, attrs: ConcatAttrs, backend: WebGPUBackend}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {axis} = attrs;

  if (inputs.length === 1) {
    return inputs[0];
  }

  const $axis = util.parseAxisParam(axis, inputs[0].shape)[0];
  const outShape =
      backend_util.computeOutShape(inputs.map(t => t.shape), $axis);
  //const $inputs = inputs.filter(t => util.sizeFromShape(t.shape) > 0);
  //if (backend.shouldExecuteOnCPU(inputs as Tensor [])) {
  //    return this.cpuBackend.concat(tensors, axis);
  //}

  // Is there a maximum number of buffers that can be uploaded to a WebGPU
  // program?
  // if (tensors.length > MAX_SSBOS_FOR_WEBGPU_PROGRAM) {
  //   const midIndex = Math.floor(tensors.length / 2);
  //   const leftSide = this.concat(tensors.slice(0, midIndex), axis);
  //   const rightSide = this.concat(tensors.slice(midIndex), axis);
  //   return this.concat([leftSide, rightSide], axis);
  // }
    //const tensors2D: Tensor2D[] = tensors.map(t => t.reshape([
    //  util.sizeFromShape(t.shape.slice(0, axis)),
    //  util.sizeFromShape(t.shape.slice(axis))
    //]));
  const tensors2D: TensorInfo[] = inputs.map(t => 
    reshape({
      inputs: {x: t},
      backend,
      attrs: {shape: [util.sizeFromShape(t.shape.slice(0, $axis)), util.sizeFromShape(t.shape.slice($axis))]}
    })
  );
  const program = new ConcatProgram((tensors2D as Tensor2D[]).map(t => t.shape));
  const res = backend.compileAndRun(program, tensors2D);
  //return res.reshape(outShape)
  return reshape({
    inputs: {x: res},
    backend,
    attrs: {shape: outShape}
  });
}

export const concatConfig: KernelConfig = {
  kernelName: Concat,
  backendName: 'webgpu',
  kernelFunc: concat as {} as KernelFunc
};
