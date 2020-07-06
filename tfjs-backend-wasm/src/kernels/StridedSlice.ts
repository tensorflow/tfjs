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

import {buffer, NamedAttrMap, NamedTensorInfoMap, registerKernel, slice_util, tensor} from '@tensorflow/tfjs-core';
import {TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface SliceInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface SliceAttrs extends NamedAttrMap {
  begin: number[];
  end: number[];
  strides: number[];
}

export function stridedSlice(
    args: {inputs: SliceInputs, attrs: SliceAttrs, backend: BackendWasm}) {
  const {inputs: {x}, attrs: {begin, end, strides}, backend} = args;
  const outShape = slice_util.computeOutShape(begin, end, strides);
  const out = backend.makeOutput(outShape, x.dtype);
  const outVals = backend.typedArrayFromHeap(out);

  if (outShape.some(axis => axis === 0)) {
    return tensor([], outShape);
  }

  const xVals = backend.typedArrayFromHeap(x);

  const outBuf = buffer(outShape, x.dtype, outVals);
  const xBuf = buffer(x.shape, x.dtype, xVals);

  for (let i = 0; i < outBuf.size; i++) {
    const loc = outBuf.indexToLoc(i);
    const newLoc: number[] = new Array(loc.length);
    for (let j = 0; j < newLoc.length; j++) {
      newLoc[j] = loc[j] * strides[j] + begin[j];
    }

    outBuf.set(xBuf.get(...newLoc), ...loc);
  }

  return out;
}

registerKernel({
  kernelName: 'StridedSlice',
  backendName: 'wasm',
  kernelFunc: stridedSlice,
});
