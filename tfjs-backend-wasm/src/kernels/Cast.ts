/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {DataType, NamedAttrMap, NamedTensorInfoMap, registerKernel, util} from '@tensorflow/tfjs-core';
import {TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface CastInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface CastAttrs extends NamedAttrMap {
  dtype: DataType;
}

function cast(
    args: {inputs: CastInputs, attrs: CastAttrs, backend: BackendWasm}) {
  const {inputs: {x}, attrs: {dtype}, backend} = args;
  const out = backend.makeOutput(x.shape, dtype);
  const {memoryOffset: inOffset} = backend.dataIdMap.get(x.dataId);
  const {memoryOffset: outOffset} = backend.dataIdMap.get(out.dataId);
  const size = util.sizeFromShape(x.shape);
  const inVals = backend.typedArrayFromHeap(inOffset, x.dtype, size);
  const outVals = backend.typedArrayFromHeap(outOffset, dtype, size);
  outVals.set(inVals);
  return out;
}

registerKernel({
  kernelName: 'Cast',
  backendName: 'wasm',
  kernelFunc: cast,
});
