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

import {KernelConfig, KernelFunc, TensorInfo, Unpack, UnpackAttrs, UnpackInputs} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {slice} from './Slice';

function unpack(
    args: {inputs: UnpackInputs, backend: BackendWasm, attrs: UnpackAttrs}):
    TensorInfo[] {
  const {inputs, backend, attrs} = args;
  const {value} = inputs;
  const {axis} = attrs;
  const numOutputs = value.shape[axis];
  const rank = value.shape.length;
  const outShape: number[] = new Array(rank - 1);
  let outIndex = 0;
  for (let i = 0; i < rank; i++) {
    if (i !== axis) {
      outShape[outIndex++] = value.shape[i];
    }
  }
  const outs: TensorInfo[] = new Array(numOutputs);
  const begin = new Array(rank).fill(0);
  const size = value.shape.slice();
  size[axis] = 1;
  for (let i = 0; i < outs.length; i++) {
    begin[axis] = i;
    outs[i] = slice({inputs: {x: value}, attrs: {begin, size}, backend});
  }
  return outs.map(({dataId, dtype}) => ({dataId, dtype, shape: outShape}));
}

export const unpackConfig: KernelConfig = {
  kernelName: Unpack,
  backendName: 'wasm',
  kernelFunc: unpack as {} as KernelFunc,
};
