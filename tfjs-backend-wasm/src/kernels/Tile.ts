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

import {KernelConfig, KernelFunc, Tile, TileAttrs, TileInputs} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

let wasmTile: (
    xId: number, xShape: Uint8Array, xShapeSize: number, newShape: Uint8Array,
    newShapeSize: number, dtype: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmTile = backend.wasm.cwrap(Tile, null /* void */, [
    'number',  // x_id
    'array',   // x_shape
    'number',  // x_shape.length
    'array',   // new_shape
    'number',  // new_shape.length
    'number'   // out_id
  ]);
}

function tile(
    args: {inputs: TileInputs, backend: BackendWasm, attrs: TileAttrs}) {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;
  const {reps} = attrs;

  const newShape: number[] = new Array(x.shape.length);
  for (let i = 0; i < newShape.length; i++) {
    newShape[i] = x.shape[i] * reps[i];
  }
  const xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);
  const newShapeBytes = new Uint8Array(new Int32Array(newShape).buffer);

  const out = backend.makeOutput(newShape, x.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmTile(
      xId, xShapeBytes, x.shape.length, newShapeBytes, newShape.length,
      CppDType[out.dtype], outId);
  return out;
}

export const tileConfig: KernelConfig = {
  kernelName: Tile,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: tile as {} as KernelFunc
};
