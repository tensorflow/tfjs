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

import {add, backend_util, FusedDepthwiseConv2D, FusedDepthwiseConv2DAttrs, FusedDepthwiseConv2DInputs, KernelConfig, Tensor} from '@tensorflow/tfjs';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';

import {depthwiseConv2dNativeImpl} from './DepthwiseConv2dNative';

export const fusedDepthwiseConv2DConfig: KernelConfig = {
  kernelName: FusedDepthwiseConv2D,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x, filter, bias, preluActivationWeights} =
        args.inputs as FusedDepthwiseConv2DInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {
      strides,
      pad,
      dilations,
      dimRoundingMode,
      activation,
      leakyreluAlpha
    } = args.attrs as {} as FusedDepthwiseConv2DAttrs;

    let $dilations = dilations;
    if ($dilations == null) {
      $dilations = [1, 1];
    }

    const convInfo = backend_util.computeConv2DInfo(
        x.shape as [number, number, number, number],
        filter.shape as [number, number, number, number], strides, $dilations,
        pad, dimRoundingMode, true /* depthwise */);

    let result = depthwiseConv2dNativeImpl(x, filter, convInfo, backend);

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
