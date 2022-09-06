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

import {FusedConv2D, FusedConv2DAttrs, FusedConv2DInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {applyActivation} from '../utils/fused_utils';
import {add} from './Add';
import {conv2D} from './Conv2D';
import {reshape} from './Reshape';

export function fusedConv2D(args: {
  inputs: FusedConv2DInputs,
  backend: MathBackendCPU,
  attrs: FusedConv2DAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, filter, bias, preluActivationWeights} = inputs;
  const {
    strides,
    pad,
    dataFormat,
    dilations,
    dimRoundingMode,
    activation,
    leakyreluAlpha
  } = attrs;

  let result = conv2D({
    inputs: {x, filter},
    backend,
    attrs: {strides, pad, dataFormat, dilations, dimRoundingMode}
  });

  if (bias) {
    const resultOld = result;
    // For NCHW format, if bias is a 1-D tensor, it is supposed to be aligned
    // to the channel of the conv2d's result; if the bias is a scalar, the
    // bias_add is computed as if the bias was broadcasted to the shape of the
    // conv2d's result.
    if (dataFormat === 'NCHW' && bias.shape.length === 1 &&
        bias.shape[0] !== 1) {
      const reshapedBias = reshape(
          {inputs: {x: bias}, backend, attrs: {shape: [bias.shape[0], 1, 1]}});
      result =
          add({inputs: {a: result, b: reshapedBias}, backend}) as TensorInfo;
      backend.disposeIntermediateTensorInfo(reshapedBias);
    } else {
      // This condition handles NHWC and NCHW (scalar case). The only other case
      // for NCHW (1D case) is handled above.
      result = add({inputs: {a: result, b: bias}, backend}) as TensorInfo;
    }
    backend.disposeIntermediateTensorInfo(resultOld);
  }

  if (activation) {
    const resultOld = result;
    // For NCHW format, if PReLu activation weights is a 1-D tensor, it is
    // supposed to be aligned with the channel of the conv2d's result. For other
    // cases, whether NCHW or NHWC data format, the conv2d result is
    // already aligned with the activation weights.
    if (dataFormat === 'NCHW' && activation === 'prelu' &&
        preluActivationWeights.shape.length === 1 &&
        preluActivationWeights.shape[0] !== 1) {
      const reshapedAlpha = reshape({
        inputs: {x: preluActivationWeights},
        backend,
        attrs: {shape: [preluActivationWeights.shape[0], 1, 1]}
      });
      result = applyActivation(
          backend, result, activation, reshapedAlpha, leakyreluAlpha);
      backend.disposeIntermediateTensorInfo(reshapedAlpha);
    } else {
      result = applyActivation(
          backend, result, activation, preluActivationWeights, leakyreluAlpha);
    }
    backend.disposeIntermediateTensorInfo(resultOld);
  }

  return result;
}

export const fusedConv2DConfig: KernelConfig = {
  kernelName: FusedConv2D,
  backendName: 'cpu',
  kernelFunc: fusedConv2D as {} as KernelFunc
};
