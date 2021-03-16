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

import {backend_util, FusedDepthwiseConv2D, FusedDepthwiseConv2DAttrs, FusedDepthwiseConv2DInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {DepthwiseConv2DProgram} from './depthwise_conv2d_webgpu';

export function fusedDepthwiseConv2D(args: {
  inputs: FusedDepthwiseConv2DInputs,
  attrs: FusedDepthwiseConv2DAttrs,
  backend: WebGPUBackend
}) {
  const {inputs, backend, attrs} = args;
  const {x, filter, bias, preluActivationWeights} = inputs;
  const {strides, pad, dilations, dimRoundingMode, activation} = attrs;

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

  const fusedActivation =
      activation ? backend.mapActivationToShaderProgram(activation) : null;
  const programInputs: TensorInfo[] = [x, filter];

  const hasBias = bias != null;
  const hasPreluActivationWeights = preluActivationWeights != null;

  if (hasBias) {
    programInputs.push(bias);
  }
  if (hasPreluActivationWeights) {
    programInputs.push(preluActivationWeights);
  }

  const program = new DepthwiseConv2DProgram(
      convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
  const dimensions = [
    convInfo.filterHeight, convInfo.filterWidth, convInfo.padInfo.top,
    convInfo.padInfo.left, convInfo.strideHeight, convInfo.strideWidth,
    convInfo.dilationHeight, convInfo.dilationWidth, convInfo.inHeight,
    convInfo.inWidth
  ];
  const uniformData = new Int32Array(dimensions);
  const result =
      backend.runWebGPUProgram(program, programInputs, 'float32', uniformData);

  return result;
}

export const fusedDepthwiseConv2DConfig: KernelConfig = {
  kernelName: FusedDepthwiseConv2D,
  backendName: 'webgpu',
  kernelFunc: fusedDepthwiseConv2D as {} as KernelFunc,
};
