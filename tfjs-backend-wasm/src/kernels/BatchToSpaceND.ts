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

import {backend_util, buffer, NamedAttrMap, NamedTensorInfoMap, registerKernel, slice_util, util} from '@tensorflow/tfjs-core';
import {TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {reshape} from './Reshape';
import {transpose} from './Transpose';

function batchToSpaceND(args) {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {blockShape, crops} = attrs;

  const prod = blockShape.reduce(function(a, b) {
    return a * b;
  });
  const reshaped = backend_util.getReshaped(x.shape, blockShape, prod);
  const permuted = backend_util.getPermuted(reshaped.length, blockShape.length);
  const reshapedPermuted =
      backend_util.getReshapedPermuted(x.shape, blockShape, prod);
  const sliceBeginCoords =
      backend_util.getSliceBeginCoords(crops, blockShape.length);
  const sliceSize =
      backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);

  const xReshaped =
      reshape({inputs: {x: x}, attrs: {shape: reshaped}, backend});
  const xReshapedT =
      transpose({inputs: {x: xReshaped}, attrs: {perm: permuted}, backend});
  const xReshapedTReshaped = reshape(
      {inputs: {x: xReshapedT}, attrs: {shape: reshapedPermuted}, backend});
  console.log(
      'batch to space', sliceBeginCoords, sliceSize, xReshapedTReshaped,
      reshapedPermuted);
  console.log(xReshaped, xReshapedT);
  return slice({
    inputs: {x: xReshapedTReshaped},
    attrs: {begin: sliceBeginCoords, size: sliceSize},
    backend
  });
}


registerKernel({
  kernelName: 'BatchToSpaceND',
  backendName: 'wasm',
  kernelFunc: batchToSpaceND
});
