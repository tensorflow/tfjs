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

import {KernelConfig, KernelFunc, Reverse, ReverseAttrs, ReverseInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {identity} from './Identity';
import {reshape} from './Reshape';

let wasmReverse: (
    xId: number, axes: Uint8Array, axesLength: number, outShape: Uint8Array,
    outShapeLength: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmReverse = backend.wasm.cwrap(Reverse, null, [
    'number',  // x_id
    'array',   // axes
    'number',  // axes_length
    'array',   // out_shape
    'number',  // out_shape_length
    'number'   // out_id
  ]);
}

export function reverse(
    args: {inputs: ReverseInputs, backend: BackendWasm, attrs: ReverseAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {dims} = attrs;

  const axes = util.parseAxisParam(dims, x.shape);

  if (x.shape.length === 0) {
    return identity({inputs: {x}, backend});
  }

  const out = backend.makeOutput(x.shape, x.dtype);
  const xId = backend.dataIdMap.get(x.dataId).id;
  const outId = backend.dataIdMap.get(out.dataId).id;

  const axesBytes = new Uint8Array(new Int32Array(axes).buffer);
  const outShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);

  wasmReverse(
      xId, axesBytes, axes.length, outShapeBytes, x.shape.length, outId);

  const reshaped =
      reshape({inputs: {x: out}, attrs: {shape: x.shape}, backend});

  backend.disposeData(out.dataId);
  return reshaped;
}

export const reverseConfig: KernelConfig = {
  kernelName: Reverse,
  backendName: 'wasm',
  kernelFunc: reverse as {} as KernelFunc,
  setupFunc: setup
};
