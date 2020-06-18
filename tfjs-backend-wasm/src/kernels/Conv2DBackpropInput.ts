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

import {Conv2DBackpropInput, Conv2DBackpropInputAttrs, Conv2DBackpropInputInputs, NamedAttrMap, NamedTensorInfoMap, registerKernel, scatter_util, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {CppDType} from './types';

let wasmConv2DBackpropInput: () => void;

function setup(backend: BackendWasm): void {
  wasmConv2DBackpropInput = backend.wasm.cwrap(Conv2DBackpropInput, null, []);
}

function conv2DBackpropInput(args: {
  backend: BackendWasm,
  inputs: NamedTensorInfoMap,
  attrs: NamedAttrMap
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {dy, filter} = inputs as Conv2DBackpropInputInputs;
  const {strides, pad, dataFormat, dimRoundingMode} =
      attrs as Conv2DBackpropInputAttrs;

  const out = backend.makeOutput();
}

registerKernel({
  kernelName: Conv2DBackpropInput,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: conv2DBackpropInput
});
