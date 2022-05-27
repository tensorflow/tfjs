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
  } else if (env().getBool('WEBGL_CONV_IM2COL')) {
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

    // If the input is a 1-D tensor, align it with the channels.
    //
    // For fusedConv2d, the inputs (x, W, bias, preluActivationWeights) are
    // supposed to be aligned with the dataFormat. The 4-D tensor inputs or
    // scalar inputs are originally aligned, but the 1-D tensor inputs are
    // supposed to be aligned with the channels (only bias and PReLU activation
    // weights could be a 1-D tensor).
    const alignInputWithDataFormat =
        (input: TensorInfo, dataFormat: 'NHWC'|'NCHW'): TensorInfo => {
          if (dataFormat === 'NCHW' && input.shape.length === 1 &&
              input.shape[0] !== 1) {
            const alignedInput = reshape({
              inputs: {x: input},
              backend,
              attrs: {shape: [input.shape[0], 1, 1]}
            });
            intermediates.push(alignedInput);
            return alignedInput;
          }
          return input;
        };

    if (hasBias) {
      inputs.push(alignInputWithDataFormat(bias, dataFormat));
    }

    if (hasPreluActivationWeights) {
      inputs.push(alignInputWithDataFormat(preluActivationWeights, dataFormat));
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
