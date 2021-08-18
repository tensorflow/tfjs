/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, Reshape, ReshapeAttrs, ReshapeInputs, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

export function reshape(
    args: {inputs: ReshapeInputs, attrs: ReshapeAttrs, backend: BackendWasm}) {
  const {inputs, attrs} = args;
  const {x} = inputs;
  const {shape} = attrs;

  const xSize = util.sizeFromShape(x.shape);
  const $shape = util.inferFromImplicitShape(shape, xSize);

  util.assert(
      xSize === util.sizeFromShape($shape),
      () => `new shape: ${$shape}, old shape: ${x.shape}. New shape and old ` +
          `shape must have the same number of elements.`);

  // Backend needs to track refCount for the dataId for reshape op
  args.backend.incRef(x.dataId);
  return {dataId: x.dataId, shape: $shape, dtype: x.dtype};
}

export const reshapeConfig: KernelConfig = {
  kernelName: Reshape,
  backendName: 'wasm',
  kernelFunc: reshape as {} as KernelFunc
};
