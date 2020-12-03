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

import {KernelConfig, KernelFunc, TensorInfo, Unpack, UnpackAttrs, UnpackInputs} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {reshape} from './Reshape';
import {slice} from './Slice';

export function unpack(
    args: {inputs: UnpackInputs, backend: MathBackendCPU, attrs: UnpackAttrs}):
    TensorInfo[] {
  const {inputs, backend, attrs} = args;
  const {value} = inputs;
  let {axis} = attrs;

  if (axis < 0) {
    axis += value.shape.length;
  }

  const valueRank = value.shape.length;

  const num = value.shape[axis];
  const outShape: number[] = new Array(valueRank - 1);
  let outIndex = 0;
  for (let i = 0; i < valueRank; i++) {
    if (i !== axis) {
      outShape[outIndex++] = value.shape[i];
    }
  }

  const begin = new Array(valueRank).fill(0);
  const size = value.shape.slice();
  size[axis] = 1;
  const res = new Array(num);
  for (let i = 0; i < res.length; i++) {
    begin[axis] = i;
    const tempRes = slice({inputs: {x: value}, backend, attrs: {begin, size}});
    res[i] = reshape({inputs: {x: tempRes}, backend, attrs: {shape: outShape}});
    backend.disposeIntermediateTensorInfo(tempRes);
  }

  return res;
}

export const unpackConfig: KernelConfig = {
  kernelName: Unpack,
  backendName: 'cpu',
  kernelFunc: unpack as {} as KernelFunc
};
