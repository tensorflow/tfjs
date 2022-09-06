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

import {KernelConfig, KernelFunc, Reshape, ReshapeAttrs, ReshapeInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

export function reshape(
    args: {inputs: ReshapeInputs, backend: WebGPUBackend, attrs: ReshapeAttrs}):
    TensorInfo {
  const {inputs, attrs} = args;
  const {x} = inputs;
  const {shape} = attrs;

  const xSize = util.sizeFromShape(x.shape);
  const $shape = util.inferFromImplicitShape(shape, xSize);
  const $xSize = util.sizeFromShape($shape);

  util.assert(
      xSize === $xSize,
      () => `The new shape (${$shape}) has ${$xSize} elements and the old ` +
          `shape (${x.shape}) has ${xSize} elements. The new shape and old ` +
          `shape must have the same number of elements.`);

  // Backend needs to track refCount for the dataId for reshape op
  args.backend.incRef(x.dataId);
  return {dataId: x.dataId, shape: $shape, dtype: x.dtype};
}

export const reshapeConfig: KernelConfig = {
  kernelName: Reshape,
  backendName: 'webgpu',
  kernelFunc: reshape as {} as KernelFunc
};
