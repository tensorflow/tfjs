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

import {backend_util, env, FusedDepthwiseConv2D, FusedDepthwiseConv2DAttrs, FusedDepthwiseConv2DInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {DepthwiseConv2DProgram} from '../conv_gpu_depthwise';
import {DepthwiseConvPacked2DProgram} from '../conv_packed_gpu_depthwise';
import {mapActivationToShaderProgram} from '../kernel_utils/kernel_funcs_utils';

export function fusedDepthwiseConv2D(args: {
  inputs: FusedDepthwiseConv2DInputs,
  attrs: FusedDepthwiseConv2DAttrs,
  backend: MathBackendWebGL
}) {
  const {inputs, backend, attrs} = args;
  const {x, filter, bias, preluActivationWeights} = inputs;
  const {strides, pad, dilations, dimRoundingMode, activation, leakyreluAlpha} =
      attrs;

  const intermediates: TensorInfo[] = [];

  let $dilations = dilations;
  if ($dilations == null) {
    $dilations = [1, 1];
  }

  util.assert(
      backend_util.eitherStridesOrDilationsAreOne(strides, $dilations),
      () => 'Error in depthwiseConv2d: Either strides or dilations must be ' +
          `1. Got strides ${strides} and dilations '${$dilations}'`);

  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number, number], strides, $dilations,
      pad, dimRoundingMode, true /* depthwise */);

  const shouldPackDepthwiseConv = env().getBool('WEBGL_PACK_DEPTHWISECONV') &&
      convInfo.strideWidth <= 2 &&
      convInfo.outChannels / convInfo.inChannels === 1;
  const fusedActivation = activation ?
      mapActivationToShaderProgram(activation, shouldPackDepthwiseConv) :
      null;
  const programInputs: TensorInfo[] = [x, filter];

  const hasBias = bias != null;
  const hasPreluActivationWeights = preluActivationWeights != null;
  const hasLeakyreluAlpha = activation === 'leakyrelu';

  if (hasBias) {
    programInputs.push(bias);
  }
  if (hasPreluActivationWeights) {
    programInputs.push(preluActivationWeights);
  }
  if (hasLeakyreluAlpha) {
    const $leakyreluAlpha = backend.makeTensorInfo(
        [], 'float32',
        util.createScalarValue(leakyreluAlpha as {} as 'float32', 'float32'));
    programInputs.push($leakyreluAlpha);
    intermediates.push($leakyreluAlpha);
  }

  let program: DepthwiseConv2DProgram|DepthwiseConvPacked2DProgram;
  if (shouldPackDepthwiseConv) {
    program = new DepthwiseConvPacked2DProgram(
        convInfo, hasBias, fusedActivation, hasPreluActivationWeights,
        hasLeakyreluAlpha);
  } else {
    program = new DepthwiseConv2DProgram(
        convInfo, hasBias, fusedActivation, hasPreluActivationWeights,
        hasLeakyreluAlpha);
  }

  const result = backend.runWebGLProgram(program, programInputs, 'float32');

  intermediates.forEach(t => backend.disposeIntermediateTensorInfo(t));

  return result;
}

export const fusedDepthwiseConv2DConfig: KernelConfig = {
  kernelName: FusedDepthwiseConv2D,
  backendName: 'webgl',
  kernelFunc: fusedDepthwiseConv2D as {} as KernelFunc,
};
