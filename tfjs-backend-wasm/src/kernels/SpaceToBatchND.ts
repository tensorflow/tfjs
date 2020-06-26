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

import {pad} from './PadV2';
import {reshape} from './Reshape';
import {transpose} from './Transpose';

function spaceToBatchND(args) {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {blockShape, paddings} = attrs;

  const prod = blockShape.reduce(function(a, b) {
    return a * b;
  });
  const completePaddings = [[0, 0]];
  completePaddings.push.apply(completePaddings, paddings);
  for (const i = 1 + blockShape.length; i < x.shape.length; ++i) {
    completePaddings.push([0, 0]);
  }

  const paddedX = pad({
    inputs: {x},
    attrs: {paddings: completePaddings, constantValue: 0},
    backend
  });
  const reshapedPaddedShape =
      backend_util.getReshaped(paddedX.shape, blockShape, prod, false);
  const permutedReshapedPaddedPermutation = backend_util.getPermuted(
      reshapedPaddedShape.length, blockShape.length, false);
  const flattenShape =
      backend_util.getReshapedPermuted(paddedX.shape, blockShape, prod, false);

  console.log(
      flattenShape, reshapedPaddedShape, permutedReshapedPaddedPermutation);
  const paddedXReshaped = reshape(
      {inputs: {x: paddedX}, attrs: {shape: reshapedPaddedShape}, backend});
  console.log('transpose input', paddedXReshaped);
  const paddedXReshapedT = transpose({
    inputs: {x: paddedXReshaped},
    attrs: {perm: permutedReshapedPaddedPermutation},
    backend
  });
  return reshape(
      {inputs: {x: paddedXReshapedT}, attrs: {shape: flattenShape}, backend});
}

registerKernel({
  kernelName: 'SpaceToBatchND',
  backendName: 'wasm',
  kernelFunc: spaceToBatchND
});
