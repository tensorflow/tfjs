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

import {backend_util, env, FusedConv2D, FusedConv2DAttrs, FusedConv2DInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {conv2dByMatMul} from './Conv2D_impl';
import {Conv2DMMVec4Program} from './conv2d_mm_vec4_webgpu';
import {Conv2DMMProgram} from './conv2d_mm_webgpu';
import {Conv2DNaiveProgram} from './conv2d_naive_webgpu';

export function fusedConv2d(args: {
  inputs: FusedConv2DInputs,
  attrs: FusedConv2DAttrs,
  backend: WebGPUBackend
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

  const hasBias = bias != null;
  const hasPreluActivationWeights = preluActivationWeights != null;

  let program: Conv2DMMProgram|Conv2DNaiveProgram|Conv2DMMVec4Program;

  if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
      convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
      convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
      (convInfo.padInfo.type === 'SAME' || convInfo.padInfo.type === 'VALID')) {
    return conv2dByMatMul({
      x,
      filter,
      convInfo,
      backend,
      bias,
      activation,
      preluActivationWeights,
      leakyreluAlpha
    });
  }

  const useNaive = env().getBool('WEBGPU_USE_NAIVE_CONV2D');

  const useVec4 =
      convInfo.inChannels % 4 === 0 && convInfo.outChannels % 4 === 0;
  const packed = !useNaive && useVec4;
  const fusedActivation = activation ?
      backend.mapActivationToShaderProgram(activation, packed) :
      null;

  if (useNaive) {
    // TODO(kainino0x): This may be obsolete, but is kept for reference.
    program = new Conv2DNaiveProgram(
        convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
  } else if (useVec4) {
    program = new Conv2DMMVec4Program(
        convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
  } else {
    program = new Conv2DMMProgram(
        convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
  }

  const padInfo = [convInfo.padInfo.top, convInfo.padInfo.left];

  const dimensions = [
    convInfo.filterHeight, convInfo.filterWidth, ...padInfo,
    convInfo.strideHeight, convInfo.strideWidth, convInfo.dilationHeight,
    convInfo.dilationWidth
  ];
  const uniformData = new Int32Array(dimensions);
  const inputVar: TensorInfo[] = [x, filter];
  if (hasBias) {
    inputVar.push(bias);
  }
  if (hasPreluActivationWeights) {
    inputVar.push(preluActivationWeights);
  }
  return backend.runWebGPUProgram(program, inputVar, x.dtype, uniformData);
}

export const fusedConv2DConfig: KernelConfig = {
  kernelName: FusedConv2D,
  backendName: 'webgpu',
  kernelFunc: fusedConv2d as {} as KernelFunc,
};
