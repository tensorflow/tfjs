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

import {backend_util, KernelConfig, KernelFunc, ReshapeAttrs, ReshapeInputs, SpaceToBatchND, SpaceToBatchNDAttrs, SpaceToBatchNDInputs, TensorInfo, TransposeAttrs, TransposeInputs, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {padV2Config} from './PadV2';
import {reshape} from './Reshape';
import {transpose} from './Transpose';

function spaceToBatchND(args: {
  inputs: SpaceToBatchNDInputs,
  backend: BackendWasm,
  attrs: SpaceToBatchNDAttrs
}) {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {blockShape, paddings} = attrs;

  const prod = util.sizeFromShape(blockShape);

  const completePaddings: Array<[number, number]> = [[0, 0]];
  completePaddings.push(...(paddings as Array<[number, number]>));

  for (let i = 1 + blockShape.length; i < x.shape.length; ++i) {
    completePaddings.push([0, 0]);
  }

  const paddedX = padV2Config.kernelFunc({
    inputs: {x},
    backend,
    attrs: {paddings: completePaddings, constantValue: 0}
  }) as TensorInfo;

  const reshapedPaddedShape =
      backend_util.getReshaped(paddedX.shape, blockShape, prod, false);

  const permutedReshapedPaddedPermutation = backend_util.getPermuted(
      reshapedPaddedShape.length, blockShape.length, false);

  const flattenShape =
      backend_util.getReshapedPermuted(paddedX.shape, blockShape, prod, false);

  const reshapeInputs: ReshapeInputs = {x: paddedX};
  const reshapeAttrs: ReshapeAttrs = {shape: reshapedPaddedShape};
  const paddedXReshaped =
      reshape({inputs: reshapeInputs, backend, attrs: reshapeAttrs});

  const transposeInputs: TransposeInputs = {x: paddedXReshaped};
  const transposeAttrs:
      TransposeAttrs = {perm: permutedReshapedPaddedPermutation};
  const paddedXT =
      transpose({inputs: transposeInputs, backend, attrs: transposeAttrs});

  const resultReshapeInputs: ReshapeInputs = {x: paddedXT};
  const resultReshapeAttrs: ReshapeAttrs = {shape: flattenShape};
  const result = reshape(
      {inputs: resultReshapeInputs, backend, attrs: resultReshapeAttrs});

  backend.disposeData(paddedX.dataId);
  backend.disposeData(paddedXReshaped.dataId);
  backend.disposeData(paddedXT.dataId);

  return result;
}

export const spaceToBatchNDConfig: KernelConfig = {
  kernelName: SpaceToBatchND,
  backendName: 'wasm',
  kernelFunc: spaceToBatchND as {} as KernelFunc
};
