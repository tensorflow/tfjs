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

import {KernelFunc, Min, MinAttrs, MinInputs, TensorInfo} from '@tensorflow/tfjs-core';
import {backend_util, KernelConfig} from '@tensorflow/tfjs-core';
import {util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {reduce} from '../kernel_utils/reduce';
import {reshape} from './Reshape';

export function min(
    args: {inputs: MinInputs, backend: WebGPUBackend, attrs: MinAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis} = attrs;
  const webgpuBackend = backend;
  const xShape = x.shape;
  const xRank = xShape.length;

  const origAxes = util.parseAxisParam(axis, xShape);
  const axes = origAxes;
  backend_util.assertAxesAreInnerMostDims('min', axes, xRank);
  const [outShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(xShape, axes);
  const reduceSize = util.sizeFromShape(reduceShape);
  const a2D = reshape(
      {inputs: {x}, attrs: {shape: [-1, reduceSize]}, backend: webgpuBackend});
  const a2DReduce = reduce(a2D, a2D.dtype, 'min', webgpuBackend);
  const reshapedOutput = reshape({
    inputs: {x: a2DReduce},
    attrs: {shape: outShape},
    backend: webgpuBackend
  });

  backend.disposeData(a2D.dataId);
  backend.disposeData(a2DReduce.dataId);

  return reshapedOutput;
}

export const minConfig: KernelConfig = {
  kernelName: Min,
  backendName: 'webgpu',
  kernelFunc: min as {} as KernelFunc
};
