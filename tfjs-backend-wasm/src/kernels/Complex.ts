/* Copyright 2020 Google LLC. All Rights Reserved.
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
 * ===========================================================================*/

import {Complex, ComplexInputs, registerKernel, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmFFT: () => void;

function setup(backend: BackendWasm): void {
  wasmFFT = backend.wasm.cwrap(Complex, null, []);
}

function fft(args: {backend: BackendWasm, inputs: ComplexInputs}): TensorInfo {
  const {backend, inputs} = args;
  const {real, imag} = inputs;
  console.log(imag);
  const out = backend.makeOutput(real.shape, 'complex64');

  wasmFFT();
  return out;
}

registerKernel({
  kernelName: Complex,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: fft
});
