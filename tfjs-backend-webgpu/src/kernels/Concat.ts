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

import {backend_util, Concat, ConcatAttrs, ConcatInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {ConcatProgram} from './concat_webgpu';
import {identity} from './Identity';
import {reshape} from './Reshape';

export function concat(
    args: {inputs: ConcatInputs, attrs: ConcatAttrs, backend: WebGPUBackend}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {axis} = attrs;

  const $axis = util.parseAxisParam(axis, inputs[0].shape)[0];
  const outShape =
      backend_util.computeOutShape(inputs.map(t => t.shape), $axis);
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

  const tensors2D: TensorInfo[] =
      $inputs.map(t => reshape({
                    inputs: {x: t},
                    backend,
                    attrs: {
                      shape: [
                        util.sizeFromShape(t.shape.slice(0, $axis)),
                        util.sizeFromShape(t.shape.slice($axis))
                      ]
                    }
                  }));
  const program =
      new ConcatProgram((tensors2D).map(t => t.shape as [number, number]));
  const res = backend.runWebGPUProgram(program, tensors2D, tensors2D[0].dtype);
  tensors2D.forEach(r => backend.disposeData(r.dataId));

  const reshapedResult =
      reshape({inputs: {x: res}, backend, attrs: {shape: outShape}});
  backend.disposeData(res.dataId);
  return reshapedResult;
}

export const concatConfig: KernelConfig = {
  kernelName: Concat,
  backendName: 'webgpu',
  kernelFunc: concat as {} as KernelFunc
};
