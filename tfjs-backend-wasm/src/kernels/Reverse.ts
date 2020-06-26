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

import {backend_util, buffer, NamedAttrMap, NamedTensorInfoMap, registerKernel, Reverse, ReverseAttrs, ReverseInputs, slice_util, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

export function reverse(args: {
  inputs: NamedTensorInfoMap,
  backend: BackendWasm,
  attrs: NamedAttrMap
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs as {} as ReverseInputs;
  const {dims} = attrs as {} as ReverseAttrs;

  const axes = util.parseAxisParam(dims, x.shape);

  const out = backend.makeOutput(x.shape, x.dtype);
  const xVals = backend.typedArrayFromHeap(x);
  const outVals = backend.typedArrayFromHeap(out);
  const outBuf = buffer(x.shape, x.dtype, outVals);

  for (let i = 0; i < outVals.length; i++) {
    const outLoc = outBuf.indexToLoc(i);
    const inLoc = outLoc.slice();
    axes.forEach(ax => inLoc[ax] = x.shape[ax] - 1 - inLoc[ax]);
    // let inPos = 0;
    outBuf.set(xVals[0], ...outLoc);
  }

  return out;
}

registerKernel({kernelName: Reverse, backendName: 'wasm', kernelFunc: reverse});
