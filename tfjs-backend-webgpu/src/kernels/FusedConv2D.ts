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

import {backend_util, env, engine, FusedConv2D, FusedConv2DAttrs, FusedConv2DInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
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
    activation
  } = attrs;

  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number, number], strides, dilations, pad,
      dimRoundingMode, false /* depthwise */, $dataFormat);

  const dataId = backend.write(null /*values*/, convInfo.outShape, x.dtype);
  const output = engine().makeTensorFromDataId(
      dataId, convInfo.outShape, x.dtype, backend);

  const hasBias = bias != null;
  const hasPreluActivationWeights = preluActivationWeights != null;
  const fusedActivation = activation ?
      backend.mapActivationToShaderProgram(activation, false) :
      null;
  let program: Conv2DMMProgram|Conv2DNaiveProgram;

  const workPerThread = env().get('WEBGPU_CONV2D_WORK_PER_THREAD') as number;
  if (workPerThread === -1) {
    // TODO(kainino0x): This may be obsolete, but is kept for reference.
    program = new Conv2DNaiveProgram(
        convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
  } else {
    program = new Conv2DMMProgram(
        convInfo, workPerThread, hasBias, fusedActivation,
        hasPreluActivationWeights);
  }

  const padInfo = [convInfo.padInfo.top, convInfo.padInfo.left];

  const dimensions = [
    convInfo.filterHeight, convInfo.filterWidth, ...padInfo,
    convInfo.strideHeight, convInfo.strideWidth, convInfo.dilationHeight,
    convInfo.dilationWidth
  ];

  const inputVar: TensorInfo[] = [x, filter];
  if (hasBias) {
    inputVar.push(bias);
  }
  if (hasPreluActivationWeights) {
    inputVar.push(preluActivationWeights);
  }
  return backend.compileAndRun(program, inputVar, output, dimensions);
}

export const fusedConv2DConfig: KernelConfig = {
  kernelName: FusedConv2D,
  backendName: 'webgpu',
  kernelFunc: fusedConv2d as {} as KernelFunc,
};
