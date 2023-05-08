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

import {add, backend_util, FusedConv2D, FusedConv2DAttrs, FusedConv2DInputs, KernelConfig, Tensor} from '@tensorflow/tfjs';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';

import {conv2dImpl} from './Conv2D';

export const fusedConv2DConfig: KernelConfig = {
  kernelName: FusedConv2D,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x, filter, bias, preluActivationWeights} =
        args.inputs as FusedConv2DInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {
      strides,
      pad,
      dataFormat,
      dilations,
      dimRoundingMode,
      activation,
      leakyreluAlpha
    } = args.attrs as unknown as FusedConv2DAttrs;

    if (dataFormat !== 'NHWC') {
      throw new Error(
          `Node backend FusedConv2D does not support dataFormat:'` +
          `${dataFormat}'. Please use 'NHWC'.`);
    }

    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(
        x.shape as [number, number, number, number],
        filter.shape as [number, number, number, number], strides, dilations,
        pad, dimRoundingMode, false /* depthwise */, $dataFormat);

    let result = conv2dImpl(x, filter, convInfo, backend);

    const toDispose = [];
    if (bias != null) {
      toDispose.push(result);
      result = add(result, bias as Tensor);
    }

    const temp = result;
    result = backend.applyActivation(
        result, activation, preluActivationWeights as Tensor, leakyreluAlpha);
    if (temp !== result) {
      toDispose.push(temp);
    }

    toDispose.forEach(t => t.dispose());
    return result;
  }
};
