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

import {backend_util, KernelConfig, KernelFunc, SpaceToBatchND, SpaceToBatchNDAttrs, SpaceToBatchNDInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';

import {padV2} from './PadV2';
import {reshape} from './Reshape';
import {transpose} from './Transpose';

export const spaceToBatchND = (args: {
  inputs: SpaceToBatchNDInputs,
  backend: MathBackendWebGL,
  attrs: SpaceToBatchNDAttrs
}): TensorInfo => {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {blockShape, paddings} = attrs;

  util.assert(
      x.shape.length <= 4,
      () => 'spaceToBatchND for rank > 4 with a WebGL backend not ' +
          'implemented yet');

  const prod = blockShape.reduce((a, b) => a * b);

  const completePaddings: Array<[number, number]> = [[0, 0]];
  completePaddings.push(...paddings as Array<[number, number]>);
  for (let i = 1 + blockShape.length; i < x.shape.length; ++i) {
    completePaddings.push([0, 0]);
  }

  const toDispose = [];

  const paddedX = padV2({
    inputs: {x},
    backend,
    attrs: {paddings: completePaddings, constantValue: 0}
  });

  const reshapedPaddedShape =
      backend_util.getReshaped(paddedX.shape, blockShape, prod, false);

  const permutedReshapedPaddedPermutation = backend_util.getPermuted(
      reshapedPaddedShape.length, blockShape.length, false);

  const flattenShape =
      backend_util.getReshapedPermuted(paddedX.shape, blockShape, prod, false);

  const reshapedPaddedX = reshape(
      {inputs: {x: paddedX}, backend, attrs: {shape: reshapedPaddedShape}});

  const paddedXT = transpose({
    inputs: {x: reshapedPaddedX},
    backend,
    attrs: {perm: permutedReshapedPaddedPermutation}
  });

  const result =
      reshape({inputs: {x: paddedXT}, backend, attrs: {shape: flattenShape}});

  toDispose.push(paddedX);
  toDispose.push(reshapedPaddedX);
  toDispose.push(paddedXT);

  toDispose.forEach(t => backend.disposeIntermediateTensorInfo(t));

  return result;
};

export const spaceToBatchNDConfig: KernelConfig = {
  kernelName: SpaceToBatchND,
  backendName: 'webgl',
  kernelFunc: spaceToBatchND as {} as KernelFunc
};
