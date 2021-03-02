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

import {KernelConfig, KernelFunc, SplitV, SplitVAttrs, SplitVInputs, util} from '@tensorflow/tfjs-core';
import {backend_util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {slice} from './Slice';

export function splitV(
    args: {inputs: SplitVInputs, attrs: SplitVAttrs, backend: BackendWasm}) {
  const {inputs, attrs, backend} = args;
  const {x} = inputs;
  const {numOrSizeSplits, axis} = attrs;

  const $axis = util.parseAxisParam(axis, x.shape)[0];

  const splitSizes = backend_util.prepareSplitSize(x, numOrSizeSplits, $axis);
  const begin = new Array(x.shape.length).fill(0);
  const size = x.shape.slice();
  return splitSizes.map(s => {
    const xSliceSize = [...size];
    xSliceSize[$axis] = s;
    const xSlice =
        slice({inputs: {x}, attrs: {begin, size: xSliceSize}, backend});
    begin[$axis] += s;
    return xSlice;
  });
}

export const splitVConfig: KernelConfig = {
  kernelName: SplitV,
  backendName: 'wasm',
  kernelFunc: splitV as {} as KernelFunc
};
