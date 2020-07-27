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

import {backend_util, SpaceToBatchND, SpaceToBatchNDAttrs, SpaceToBatchNDInputs, TensorInfo} from '@tensorflow/tfjs-core';
import {KernelConfig} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

import {padV2Config} from './PadV2';
import {reshapeConfig} from './Reshape';
import {transposeConfig} from './Transpose';

export const spaceToBatchNDConfig: KernelConfig = {
  kernelName: SpaceToBatchND,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {x} = inputs as SpaceToBatchNDInputs;
    const {blockShape, paddings} = attrs as {} as SpaceToBatchNDAttrs;
    const cpuBackend = backend as MathBackendCPU;

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
      attrs: {completePaddings, constantValue: 0}
    }) as TensorInfo;

    const reshapedPaddedShape =
        backend_util.getReshaped(paddedX.shape, blockShape, prod, false);

    const permutedReshapedPaddedPermutation = backend_util.getPermuted(
        reshapedPaddedShape.length, blockShape.length, false);

    const flattenShape = backend_util.getReshapedPermuted(
        paddedX.shape, blockShape, prod, false);

    const paddedXReshaped = reshapeConfig.kernelFunc({
      inputs: {x: paddedX},
      backend,
      attrs: {shape: reshapedPaddedShape}
    }) as TensorInfo;

    const paddedXT = transposeConfig.kernelFunc({
      inputs: {x: paddedXReshaped},
      backend,
      attrs: {perm: permutedReshapedPaddedPermutation}
    }) as TensorInfo;

    const result = reshapeConfig.kernelFunc(
        {inputs: {x: paddedXT}, backend, attrs: {shape: flattenShape}});

    cpuBackend.disposeData(paddedX.dataId);
    cpuBackend.disposeData(paddedXReshaped.dataId);
    cpuBackend.disposeData(paddedXT.dataId);

    return result;
  }
};
