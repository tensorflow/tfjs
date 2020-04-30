/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

interface SplitInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface SplitAttrs extends NamedAttrMap {
  numOrSizeSplits: number[];
  axis: number;
}

export function split(
    args: {inputs: SplitInputs, attrs: SplitAttrs, backend: BackendWasm}) {
  const {inputs: {x}, attrs: {numOrSizeSplits, axis}, backend} = args;
  const out = backend.makeOutput(x.shape, x.dtype);
  console.log(numOrSizeSplits, axis);
  return out;
}

registerKernel({kernelName: 'SplitV', backendName: 'wasm', kernelFunc: split});
