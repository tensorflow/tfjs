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
import {ArgMaxAttrs, ArgMaxInputs, ArgMinAttrs, ArgMinInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {permuteAxesAndTranspose} from './kernel_utils';
import {CppDType} from './types';

export function createArgMinMaxKernelConfig(kernelName: 'ArgMin'|
                                            'ArgMax'): KernelConfig {
  let wasmFunc: (
      xId: number, dtype: number, outerSize: number, innerSize: number,
      outId: number) => void;

  function setupFunc(backend: BackendWasm) {
    wasmFunc = backend.wasm.cwrap(kernelName, null /* void */, [
      'number',  // x_id
      'number',  // dtype
      'number',  // outer_size
      'number',  // inner_size
      'number'   // out_id
    ]);
  }

  function kernelFunc(args: {
    backend: BackendWasm,
    inputs: ArgMinInputs&ArgMaxInputs,
    attrs: ArgMinAttrs&ArgMaxAttrs,
  }): TensorInfo {
    const {backend, inputs, attrs} = args;
    const {axis} = attrs;
    const {x} = inputs;
    const xId = backend.dataIdMap.get(x.dataId).id;
    let inputId = xId;
    let input = x;

    const {transposed, axes, inputWasTransposed} =
        permuteAxesAndTranspose(x, axis, backend);

    if (inputWasTransposed) {
      const transposedId = backend.dataIdMap.get(transposed.dataId).id;
      if (transposedId !== xId) {
        // transpose was not a no-op. We will need to dispose of this
        // once we are done.
        input = transposed;
        inputId = transposedId;
      }
    }

    const outShape = input.shape.slice(0, -1);
    const out = backend.makeOutput(outShape, 'int32');
    const outId = backend.dataIdMap.get(out.dataId).id;
    const outerSize = util.sizeFromShape(out.shape);
    const innerSize = input.shape[axes[0]];
    wasmFunc(inputId, CppDType[input.dtype], outerSize, innerSize, outId);

    if (inputWasTransposed) {
      // dispose of the transposed tensor.
      backend.disposeData(transposed.dataId);
    }

    return out;
  }

  return {
    kernelName,
    backendName: 'wasm',
    setupFunc,
    kernelFunc: kernelFunc as unknown as KernelFunc,
  };
}
