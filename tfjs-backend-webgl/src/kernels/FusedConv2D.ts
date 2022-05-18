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

import {backend_util, env, FusedConv2D, FusedConv2DAttrs, FusedConv2DInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {Conv2DProgram} from '../conv_gpu';
import {mapActivationToShaderProgram} from '../kernel_utils/kernel_funcs_utils';

import {conv2dByMatMul, conv2dWithIm2Row} from './Conv2D_impl';
import {reshape} from './Reshape';

export function fusedConv2d(args: {
  inputs: FusedConv2DInputs,
  attrs: FusedConv2DAttrs,
  backend: MathBackendWebGL
}) {
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

  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number, number], strides, dilations, pad,
      dimRoundingMode, false /* depthwise */, $dataFormat);
  let out: TensorInfo;
  const intermediates: TensorInfo[] = [];

  if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
      convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
      convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
      (convInfo.padInfo.type === 'SAME' || convInfo.padInfo.type === 'VALID')) {
    out = conv2dByMatMul({
      x,
      filter,
      convInfo,
      backend,
      bias,
      activation,
      preluActivationWeights,
      leakyreluAlpha
    });
  } else if (env().getBool('WEBGL_CONV_IM2COL') && x.shape[0] === 1) {
    out = conv2dWithIm2Row({
      x,
      filter,
      convInfo,
      backend,
      bias,
      activation,
      preluActivationWeights,
      leakyreluAlpha
    });
  } else {
    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    const hasLeakyreluAlpha = activation === 'leakyrelu';
    const fusedActivation =
        activation ? mapActivationToShaderProgram(activation, false) : null;
    const program = new Conv2DProgram(
        convInfo, hasBias, fusedActivation, hasPreluActivationWeights,
        hasLeakyreluAlpha);
    const inputs: TensorInfo[] = [x, filter];

    if (bias) {
      // For NCHW format, if bias is a 1-D tensor, it is supposed to be
      // aligned with the channel of the conv2d's result; if the bias is a
      // scalar, the bias_add is computed as if the bias was broadcasted to the
      // shape of the conv2d's result.
      if ($dataFormat === 'channelsFirst' && bias.shape.length === 1 &&
          bias.shape[0] !== 1) {
        const alignedBias = reshape({
          inputs: {x: bias},
          backend,
          attrs: {shape: [bias.shape[0], 1, 1]}
        });
        inputs.push(alignedBias);
        intermediates.push(alignedBias);
      } else {
        inputs.push(bias);
      }
    }

    if (preluActivationWeights) {
      // For NCHW format, if PReLu activation weights is a 1-D tensor, it is
      // supposed to be aligned with the channel of the conv2d's result. For
      // other cases, whether NCHW or NHWC data format, the conv2d result is
      // already aligned with the activation weights.
      if ($dataFormat === 'channelsFirst' &&
          preluActivationWeights.shape.length === 1 &&
          preluActivationWeights.shape[0] !== 1) {
        const alignedPreluActivationWeights = reshape({
          inputs: {x: preluActivationWeights},
          backend,
          attrs: {shape: [preluActivationWeights.shape[0], 1, 1]}
        });
        inputs.push(alignedPreluActivationWeights);
        intermediates.push(alignedPreluActivationWeights);
      } else {
        inputs.push(preluActivationWeights);
      }
    }

    if (hasLeakyreluAlpha) {
      const $leakyreluAlpha = backend.makeTensorInfo(
          [], 'float32',
          util.createScalarValue(leakyreluAlpha as {} as 'float32', 'float32'));
      inputs.push($leakyreluAlpha);
      intermediates.push($leakyreluAlpha);
    }
    out = backend.runWebGLProgram(program, inputs, 'float32');
  }

  const outReshaped =
      reshape({inputs: {x: out}, backend, attrs: {shape: convInfo.outShape}});

  intermediates.push(out);
  intermediates.forEach(t => backend.disposeIntermediateTensorInfo(t));

  return outReshaped;
}

export const fusedConv2DConfig: KernelConfig = {
  kernelName: FusedConv2D,
  backendName: 'webgl',
  kernelFunc: fusedConv2d as {} as KernelFunc,
};
