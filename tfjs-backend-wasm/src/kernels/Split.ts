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

import {NamedAttrMap, NamedTensorInfoMap, registerKernel, util} from '@tensorflow/tfjs-core';

import {TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {slice} from './Slice';

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

  const $axis = util.parseAxisParam(axis, x.shape)[0];

  let splitSizes: number[];
  if (typeof (numOrSizeSplits) === 'number') {
    splitSizes =
        new Array(numOrSizeSplits).fill(x.shape[$axis] / numOrSizeSplits);
  } else {
    splitSizes = numOrSizeSplits;
  }

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

registerKernel({kernelName: 'SplitV', backendName: 'wasm', kernelFunc: split});
