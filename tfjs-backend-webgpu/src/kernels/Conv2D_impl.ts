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

import {backend_util, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {Conv2DMMVec4Program} from '../conv2d_mm_vec4_webgpu';
import {Conv2DMMProgram} from '../conv2d_mm_webgpu';

import {batchMatMulImpl} from './BatchMatMul_impl';
import {reshape} from './Reshape';

type Conv2DConfig = {
  x: TensorInfo,
  filter: TensorInfo,
  convInfo: backend_util.Conv2DInfo,
  backend: WebGPUBackend,
  bias?: TensorInfo,
  preluActivationWeights?: TensorInfo,
  leakyreluAlpha?: number,
  activation?: backend_util.Activation
};

// For 1x1 kernels that iterate through every point in the input, convolution
// can be expressed as matrix multiplication (without need for memory
// remapping).
function conv2dByMatMul({
  x,
  filter,
  convInfo,
  backend,
  bias = null,
  preluActivationWeights = null,
  leakyreluAlpha = 0,
  activation = null
}: Conv2DConfig) {
  const isChannelsLast = convInfo.dataFormat === 'channelsLast';
  const transposeA = isChannelsLast ? false : true;
  const transposeB = false;

  const sameSize = isChannelsLast &&
      convInfo.filterHeight === convInfo.inHeight &&
      convInfo.filterWidth === convInfo.inWidth &&
      convInfo.padInfo.type === 'VALID';
  let xReshaped;
  let filterReshaped;

  if (sameSize) {
    const sharedDim =
        convInfo.inHeight * convInfo.inWidth * convInfo.inChannels;
    xReshaped = reshape({
      inputs: {x},
      backend,
      attrs: {shape: [1, convInfo.batchSize, sharedDim]}
    });
    filterReshaped = reshape({
      inputs: {x: filter},
      backend,
      attrs: {shape: [1, sharedDim, convInfo.outChannels]}
    });
  } else {
    xReshaped = reshape({
      inputs: {x},
      backend,
      attrs: {
        shape: isChannelsLast ?
            [
              convInfo.batchSize, convInfo.inHeight * convInfo.inWidth,
              convInfo.inChannels
            ] :
            [
              convInfo.batchSize, convInfo.inChannels,
              convInfo.inHeight * convInfo.inWidth
            ]
      }
    });
    filterReshaped = reshape({
      inputs: {x: filter},
      backend,
      attrs: {shape: [1, convInfo.inChannels, convInfo.outChannels]}
    });
  }

  const result = batchMatMulImpl({
    a: isChannelsLast ? xReshaped : filterReshaped,
    b: isChannelsLast ? filterReshaped : xReshaped,
    transposeA,
    transposeB,
    backend,
    bias,
    activation,
    preluActivationWeights,
    leakyreluAlpha
  });
  const out = reshape(
      {inputs: {x: result}, backend, attrs: {shape: convInfo.outShape}});

  backend.disposeData(xReshaped.dataId);
  backend.disposeData(filterReshaped.dataId);
  backend.disposeData(result.dataId);

  return out;
}

export function conv2DImpl({
  x,
  filter,
  convInfo,
  backend,
  bias = null,
  preluActivationWeights = null,
  leakyreluAlpha = 0,
  activation = null
}: Conv2DConfig) {
  const hasBias = bias != null;
  const hasPreluActivationWeights = preluActivationWeights != null;
  const isChannelsLast = convInfo.dataFormat === 'channelsLast';
  let program: Conv2DMMProgram|Conv2DMMVec4Program;
  const sameSize = isChannelsLast &&
      convInfo.filterHeight === convInfo.inHeight &&
      convInfo.filterWidth === convInfo.inWidth &&
      convInfo.padInfo.type === 'VALID';
  if (sameSize ||
      (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
       convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
       convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
       (convInfo.padInfo.type === 'SAME' ||
        convInfo.padInfo.type === 'VALID'))) {
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

  const useVec4 =
      (convInfo.inChannels % 4 === 0 ||
       (convInfo.inChannels === 3 && convInfo.padInfo.type === 'VALID')) &&
      convInfo.outChannels % 4 === 0 && isChannelsLast;

  const padInfo = [convInfo.padInfo.top, convInfo.padInfo.left];
  const dimensions = [
    {type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth]},
    {type: 'int32', data: [...padInfo]},
    {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]},
    {type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth]}
  ];

  if (useVec4) {
    program = new Conv2DMMVec4Program(
        convInfo, hasBias, activation, hasPreluActivationWeights);
  } else {
    program = new Conv2DMMProgram(
        convInfo, hasBias, activation, hasPreluActivationWeights);
  }

  const dimAOuter = convInfo.outHeight * convInfo.outWidth;
  const dimBOuter = convInfo.outChannels;
  const dimInner =
      convInfo.filterHeight * convInfo.filterWidth * convInfo.inChannels;
  dimensions.push(
      {type: 'int32', data: [dimAOuter]}, {type: 'int32', data: [dimBOuter]},
      {type: 'int32', data: [dimInner]});

  const inputVar: TensorInfo[] = [x, filter];
  if (hasBias) {
    inputVar.push(bias);
  }
  if (hasPreluActivationWeights) {
    inputVar.push(preluActivationWeights);
  }
  if (activation === 'leakyrelu') {
    dimensions.push({type: 'float32', data: [leakyreluAlpha]});
    program.uniforms += ' alpha : f32,';
  }
  return backend.runWebGPUProgram(program, inputVar, x.dtype, dimensions);
}
