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

import {backend_util, KernelConfig, KernelFunc, NamedAttrMap, ReshapeAttrs, ReshapeInputs, SpaceToBatchND, SpaceToBatchNDAttrs, SpaceToBatchNDInputs, TensorInfo, TransposeAttrs, TransposeInputs} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

import {padV2Config} from './PadV2';
import {reshapeConfig} from './Reshape';
import {transposeConfig} from './Transpose';

function spaceToBatchND(args: {
  inputs: SpaceToBatchNDInputs,
  backend: MathBackendCPU,
  attrs: SpaceToBatchNDAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {blockShape, paddings} = attrs;

  assertNotComplex([x], 'spaceToBatchND');

  const prod = blockShape.reduce((a, b) => a * b);

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
  const paddedXReshaped = reshapeConfig.kernelFunc({
    inputs: reshapeInputs,
    backend,
    attrs: reshapeAttrs as {} as NamedAttrMap
  }) as TensorInfo;

  const transposeInputs: TransposeInputs = {x: paddedXReshaped};
  const transposeAttrs:
      TransposeAttrs = {perm: permutedReshapedPaddedPermutation};
  const paddedXT = transposeConfig.kernelFunc({
    inputs: transposeInputs,
    backend,
    attrs: transposeAttrs as {} as NamedAttrMap
  }) as TensorInfo;

  const resultReshapeInputs: ReshapeInputs = {x: paddedXT};
  const resultReshapeAttrs: ReshapeAttrs = {shape: flattenShape};
  const result = reshapeConfig.kernelFunc({
    inputs: resultReshapeInputs,
    backend,
    attrs: resultReshapeAttrs as {} as NamedAttrMap
  });

  backend.disposeInternal(paddedX.dataId);
  backend.disposeInternal(paddedXReshaped.dataId);
  backend.disposeInternal(paddedXT.dataId);

  return result as TensorInfo;
}

export const spaceToBatchNDConfig: KernelConfig = {
  kernelName: SpaceToBatchND,
  backendName: 'cpu',
  kernelFunc: spaceToBatchND as {} as KernelFunc
};
