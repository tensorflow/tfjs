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

import {BroadcastTo, BroadCastToAttrs, BroadcastToInputs, NamedAttrMap, NamedTensorInfoMap, registerKernel} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {tile} from './Tile';

function broadcastTo(args: {
  inputs: NamedTensorInfoMap,
  attrs: NamedAttrMap,
  backend: BackendWasm
}) {
  const {inputs, backend, attrs} = args;
  const {x} = inputs as {} as BroadcastToInputs;
  const {shape, inputShape} = attrs as {} as BroadCastToAttrs;
  const reps = Array.from(shape);
  for (let i = shape.length - 1; i >= 0; i--) {
    if (inputShape[i] === shape[i]) {
      reps[i] = 1;
    } else if (x.shape[i] !== 1) {
      throw new Error(
          `broadcastTo(): [${x.shape}] cannot be broadcast to [${shape}].`);
    }
  }

  return tile({backend, inputs: {x}, attrs: {reps}});
}

registerKernel(
    {kernelName: BroadcastTo, backendName: 'wasm', kernelFunc: broadcastTo});
