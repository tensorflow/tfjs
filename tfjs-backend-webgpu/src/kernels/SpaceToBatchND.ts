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

import {backend_util, KernelConfig, KernelFunc, SpaceToBatchND, SpaceToBatchNDAttrs, SpaceToBatchNDInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {SpaceToBatchNDProgram} from '../space_to_batchND_webgpu';

import {reshape} from './Reshape';

export const spaceToBatchND = (args: {
  inputs: SpaceToBatchNDInputs,
  backend: WebGPUBackend,
  attrs: SpaceToBatchNDAttrs
}): TensorInfo => {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {blockShape, paddings} = attrs;

  util.assert(
      x.shape.length <= 4,
      () => 'spaceToBatchND for rank > 4 with a WebGPU backend not ' +
          'implemented yet');

  const prod = blockShape.reduce((a, b) => a * b);

  const completePaddings: Array<[number, number]> = [[0, 0]];
  completePaddings.push(...paddings as Array<[number, number]>);
  for (let i = 1 + blockShape.length; i < x.shape.length; ++i) {
    completePaddings.push([0, 0]);
  }

  const paddedXShape = completePaddings.map(
      (p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
  const reshapedPaddedShape =
      backend_util.getReshaped(paddedXShape, blockShape, prod, false);

  const permutedReshapedPaddedPermutation = backend_util.getPermuted(
      reshapedPaddedShape.length, blockShape.length, false);

  const flattenShape =
      backend_util.getReshapedPermuted(paddedXShape, blockShape, prod, false);

  const paddedXShapeStrides = util.computeStrides(paddedXShape);
  const program = new SpaceToBatchNDProgram(
      x.shape, paddedXShape, completePaddings, reshapedPaddedShape,
      permutedReshapedPaddedPermutation, paddedXShapeStrides.length);
  const uniformData = [
    {type: 'int32', data: reshapedPaddedShape},
    {type: 'int32', data: paddedXShapeStrides}
  ];
  completePaddings.map(
      p => uniformData.push({type: 'int32', data: [p[0], p[1]]}));
  const paddedXT = backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
  const result =
      reshape({inputs: {x: paddedXT}, backend, attrs: {shape: flattenShape}});
  backend.disposeData(paddedXT.dataId);
  return result;
};

export const spaceToBatchNDConfig: KernelConfig = {
  kernelName: SpaceToBatchND,
  backendName: 'webgpu',
  kernelFunc: spaceToBatchND as unknown as KernelFunc
};
