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

import {KernelConfig, KernelFunc, Step, StepAttrs, StepInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

let wasmStep: (xId: number, alpha: number, dtype: number, outId: number) =>
    void;

function setup(backend: BackendWasm): void {
  wasmStep = backend.wasm.cwrap(Step, null /*void*/, [
    'number',  // x_id
    'number',  // alpha
    'number',  // dtype
    'number',  // out_id
  ]);
}

function step(
    args: {backend: BackendWasm, inputs: StepInputs, attrs: StepAttrs}):
    TensorInfo {
  const {backend, inputs, attrs} = args;
  const {alpha} = attrs;
  const {x} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;

  const out = backend.makeOutput(x.shape, x.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmStep(xId, alpha, CppDType[x.dtype], outId);
  return out;
}

export const stepConfig: KernelConfig = {
  kernelName: Step,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: step as {} as KernelFunc
};
