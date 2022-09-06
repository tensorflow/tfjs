/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {backend_util, BatchToSpaceND, BatchToSpaceNDAttrs, BatchToSpaceNDInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {reshape} from './Reshape';
import {slice} from './Slice';
import {transpose} from './Transpose';

function batchToSpaceND(args: {
  inputs: BatchToSpaceNDInputs,
  backend: BackendWasm,
  attrs: BatchToSpaceNDAttrs
}) {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {blockShape, crops} = attrs;

  const prod = blockShape.reduce((a, b) => a * b);

  const reshaped = backend_util.getReshaped(x.shape, blockShape, prod);
  const permuted = backend_util.getPermuted(reshaped.length, blockShape.length);
  const reshapedPermuted =
      backend_util.getReshapedPermuted(x.shape, blockShape, prod);
  const sliceBeginCoords =
      backend_util.getSliceBeginCoords(crops, blockShape.length);
  const sliceSize =
      backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);

  const xReshaped = reshape({inputs: {x}, backend, attrs: {shape: reshaped}});
  const xTransposed =
      transpose({inputs: {x: xReshaped}, backend, attrs: {perm: permuted}});
  const xTransposedReshaped = reshape(
      {inputs: {x: xTransposed}, backend, attrs: {shape: reshapedPermuted}});
  const result = slice({
    inputs: {x: xTransposedReshaped},
    backend,
    attrs: {begin: sliceBeginCoords, size: sliceSize}
  });

  backend.disposeData(xReshaped.dataId);
  backend.disposeData(xTransposed.dataId);
  backend.disposeData(xReshaped.dataId);

  return result;
}

export const batchToSpaceNDConfig: KernelConfig = {
  kernelName: BatchToSpaceND,
  backendName: 'wasm',
  kernelFunc: batchToSpaceND as {} as KernelFunc
};
