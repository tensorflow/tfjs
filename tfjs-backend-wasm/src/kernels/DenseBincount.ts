/**
 * @license
 * Copyright 2023 Google LLC.
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
import {DenseBincount, DenseBincountAttrs, DenseBincountInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

let wasmDenseBincount: (
    xId: number, xShape: Uint8Array, xShapeLen: number, size: number,
    hasWeights: boolean, weightsId: number, weightsDType: CppDType,
    binaryOutput: boolean, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmDenseBincount = backend.wasm.cwrap('DenseBincount', null /*void*/, [
    'number',   // xId
    'array',    // xShape
    'number',   // xShapeLen
    'number',   // size
    'boolean',  // hasWeights
    'number',   // weightsId
    'number',   // weightsDType
    'boolean',  // binaryOutput
    'number',   // outId
  ]);
}

function denseBincount(args: {
  backend: BackendWasm,
  inputs: DenseBincountInputs,
  attrs: DenseBincountAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {x, weights} = inputs;
  const {size, binaryOutput} = attrs;

  const hasWeights = weights.shape.reduce((p, v) => p * v, 1) !== 0;
  const outShape = x.shape.length === 1 ? [size] : [x.shape[0], size];
  const out = backend.makeOutput(outShape, weights.dtype);

  function tensorId(x: TensorInfo) {
    return backend.dataIdMap.get(x.dataId).id;
  }
  wasmDenseBincount(
      tensorId(x), new Uint8Array(new Int32Array(x.shape).buffer),
      x.shape.length, size, hasWeights, tensorId(weights),
      CppDType[weights.dtype], binaryOutput, tensorId(out));

  return out;
}

export const denseBincountConfig: KernelConfig = {
  kernelName: DenseBincount,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: denseBincount as unknown as KernelFunc
};
