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

import {backend_util, BinaryInputs, DataType, KernelConfig, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

export function createBinaryKernelConfig(
    kernelName: string, supportsFullBroadcast: boolean,
    dtype?: DataType): KernelConfig {
  let wasmFunc:
      (aId: number, aShape: Uint8Array, aShapeLen: number, bId: number,
       bShape: Uint8Array, bShapeLen: number, dtype: number, outId: number) =>
          void;

  function setupFunc(backend: BackendWasm): void {
    wasmFunc = backend.wasm.cwrap(kernelName, null /* void */, [
      'number',  // a_id,
      'array',   // a_shape
      'number',  // a_shape.length
      'number',  // b_id
      'array',   // b_shape
      'number',  // b_shape.length
      'number',  // dtype
      'number'   // out_id
    ]);
  }

  function kernelFunc(args: {backend: BackendWasm, inputs: BinaryInputs}):
      TensorInfo {
    const {backend, inputs} = args;
    const {a, b} = inputs;
    const aId = backend.dataIdMap.get(a.dataId).id;
    const bId = backend.dataIdMap.get(b.dataId).id;

    const outputType = dtype != null ? dtype : a.dtype;
    const newShape = backend_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const out = backend.makeOutput(newShape, outputType);

    // Short-circuit zero-sized tensors.
    if (util.sizeFromShape(newShape) === 0) {
      return out;
    }

    const aShapeBytes = new Uint8Array(new Int32Array(a.shape).buffer);
    const bShapeBytes = new Uint8Array(new Int32Array(b.shape).buffer);
    const outId = backend.dataIdMap.get(out.dataId).id;
    const kernelFunc = () => wasmFunc(
        aId, aShapeBytes, a.shape.length, bId, bShapeBytes, b.shape.length,
        CppDType[a.dtype], outId);

    kernelFunc();
    return out;
  }

  return {kernelName, backendName: 'wasm', setupFunc, kernelFunc};
}
