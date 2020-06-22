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

import {FFT, FFTInputs, registerKernel, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {complex} from './Complex';
// import {CppDType} from './types';

let wasmFFT: (
    inputId: number, imagInputId: number, outerDim: number, innerDim: number,
    isRealComponent: number, outputId: number) => void;

function setup(backend: BackendWasm): void {
  wasmFFT = backend.wasm.cwrap(FFT, null, [
    'number',  // inputId
    'number',  // imagInputId
    'number',  // outerDim
    'number',  // innerDim
    'number',  // isRealComponent
    'number',  // outputId
  ]);
}

function fft(args: {backend: BackendWasm, inputs: FFTInputs}): TensorInfo {
  const {backend, inputs} = args;
  const {input} = inputs;
  const inputData = backend.dataIdMap.get(input.dataId);
  const realInput = inputData.complexTensors.real;
  const imagInput = inputData.complexTensors.imag;
  const realInputId = backend.dataIdMap.get(realInput.dataId).id;
  const imagInputId = backend.dataIdMap.get(imagInput.dataId).id;

  const real = backend.makeOutput(realInput.shape, realInput.dtype);
  const imag = backend.makeOutput(imagInput.shape, imagInput.dtype);
  const realId = backend.dataIdMap.get(real.dataId).id;
  const imagId = backend.dataIdMap.get(imag.dataId).id;

  const [outerDim, innerDim] = input.shape;

  wasmFFT(
      realInputId, imagInputId, outerDim, innerDim, 1 /* is real component */,
      realId);
  wasmFFT(
      realInputId, imagInputId, outerDim, innerDim,
      0 /* is not real component */, imagId);

  const out = complex({backend, inputs: {real, imag}});
  return out;
}

registerKernel(
    {kernelName: FFT, backendName: 'wasm', setupFunc: setup, kernelFunc: fft});
