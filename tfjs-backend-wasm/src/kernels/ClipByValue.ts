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

import {NamedAttrMap, NamedTensorInfoMap, registerKernel} from '@tensorflow/tfjs-core';
import {TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface ClipByValueInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface ClipByValueAttrs extends NamedAttrMap {
  min: number;
  max: number;
}

let wasmClip: (xId: number, min: number, max: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmClip = backend.wasm.cwrap('ClipByValue', null /* void */, [
    'number',  // x_id
    'number',  // min
    'number',  // max
    'number'   // out_id
  ]);
}

function clip(args: {
  inputs: ClipByValueInputs,
  backend: BackendWasm,
  attrs: ClipByValueAttrs
}) {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {min, max} = attrs;
  const xId = backend.dataIdMap.get(x.dataId).id;
  const out = backend.makeOutput(x.shape, 'float32');
  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmClip(xId, min, max, outId);
  return out;
}

registerKernel({
  kernelName: 'ClipByValue',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: clip
});
