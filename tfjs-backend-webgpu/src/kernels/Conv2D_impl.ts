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

import {backend_util, env, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {Conv2DMMProgram} from '../conv2d_mm_webgpu';
import {Conv2DNaiveProgram} from '../conv2d_naive_webgpu';
import {Im2ColProgram} from '../im2col_webgpu';
import {WebGPUProgram} from '../webgpu_program';

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

// conv2dByMatMul fuses height and width into one dimension to compute
// batchMatMul, so bias and activation weights are also supposed to fuse the two
// dimensions into one.
//
// This function computes the target shape for fusing height and width
// dimensions. Returning null means the shape is already compatible.
function getShapeForBatchMatMul(
    shape: number[], isChannelsLast: boolean): number[] {
  const length = shape.length;
  if (length >= 3) {
    return isChannelsLast ?
        [
          ...shape.slice(0, -3) /* batch */,
          shape[length - 3] * shape[length - 2] /* height * width */,
          shape[length - 1] /* channel */
        ] :
        [
          ...shape.slice(0, -3) /* batch */, shape[length - 3] /* channel */,
          shape[length - 2] * shape[length - 1] /* height * width */
        ];
  } else if (!isChannelsLast && length === 1 && shape[0] > 1) {
    return [shape[0], 1];
  } else {
    return null;
  }
}

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
  const intermediates: TensorInfo[] = [];
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
  intermediates.push(xReshaped);
  intermediates.push(filterReshaped);

  if (preluActivationWeights != null) {
    const targetShape =
        getShapeForBatchMatMul(preluActivationWeights.shape, isChannelsLast);
    if (targetShape != null) {
      preluActivationWeights = reshape({
        inputs: {x: preluActivationWeights},
        backend,
        attrs: {shape: targetShape}
      });
      intermediates.push(preluActivationWeights);
    }
  }

  if (bias != null) {
    const targetShape = getShapeForBatchMatMul(bias.shape, isChannelsLast);
    if (targetShape != null) {
      bias = reshape({inputs: {x: bias}, backend, attrs: {shape: targetShape}});
      intermediates.push(bias);
    }
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
  intermediates.push(result);

  for (const i of intermediates) {
    backend.disposeData(i.dataId);
  }

  return out;
}

// Implements the im2col algorithm as outlined in "High Performance
// Convolutional Neural Networks for Document Processing" (Suvisoft, 2006)
function conv2dWithIm2Col({
  x,
  filter,
  convInfo,
  backend,
  bias = null,
  preluActivationWeights = null,
  leakyreluAlpha = 0,
  activation = null
}: Conv2DConfig) {
  // Rearranges conv2d input so each block to be convolved over forms the
  // row of a new matrix with shape [outHeight * outWidth,
  // filterWidth * filterHeight * inChannels]. The filter is also rearranged so
  // each output channel forms a col of a new matrix with shape [
  // filterWidth * filterHeight * inChannels, outChannels]. The convolution is
  // then computed by multiplying these matrices and reshaping the result.
  const {
    filterWidth,
    filterHeight,
    inChannels,
    strideWidth,
    strideHeight,
    padInfo,
    outWidth,
    outHeight,
    dilationWidth,
    dilationHeight,
    dataFormat
  } = convInfo;

  const isChannelsLast = dataFormat === 'channelsLast';

  const sharedDim = filterWidth * filterHeight * inChannels;
  const numCols = outHeight * outWidth;
  const x2ColShape = isChannelsLast ? [convInfo.batchSize, numCols, sharedDim] :
                                      [convInfo.batchSize, sharedDim, numCols];

  const im2ColProgram = new Im2ColProgram(x2ColShape, isChannelsLast);
  const dimensions = [
    {type: 'int32', data: [padInfo.top, padInfo.left]},      // Padding.
    {type: 'int32', data: [strideHeight, strideWidth]},      // Stride.
    {type: 'int32', data: [dilationHeight, dilationWidth]},  // Dilation.
    {type: 'int32', data: [outWidth]},
    {type: 'int32', data: [inChannels * filterWidth]},  // itemsPerBlockRow.
    {type: 'int32', data: [inChannels]}
  ];
  const x2Col =
      backend.runWebGPUProgram(im2ColProgram, [x], x.dtype, dimensions);

  const intermediates: TensorInfo[] = [];
  intermediates.push(x2Col);

  const filterReshaped = reshape(
      {inputs: {x: filter}, backend, attrs: {shape: [1, sharedDim, -1]}});
  intermediates.push(filterReshaped);

  if (preluActivationWeights != null) {
    const targetShape =
        getShapeForBatchMatMul(preluActivationWeights.shape, isChannelsLast);
    if (targetShape != null) {
      preluActivationWeights = reshape({
        inputs: {x: preluActivationWeights},
        backend,
        attrs: {shape: targetShape}
      });
      intermediates.push(preluActivationWeights);
    }
  }

  if (bias != null) {
    const targetShape = getShapeForBatchMatMul(bias.shape, isChannelsLast);
    if (targetShape != null) {
      bias = reshape({inputs: {x: bias}, backend, attrs: {shape: targetShape}});
      intermediates.push(bias);
    }
  }

  const transposeA = isChannelsLast ? false : true;
  const transposeB = false;
  const result = batchMatMulImpl({
    a: isChannelsLast ? x2Col : filterReshaped,
    b: isChannelsLast ? filterReshaped : x2Col,
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
  intermediates.push(result);
  for (const i of intermediates) {
    backend.disposeData(i.dataId);
  }

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
  const sameSize = isChannelsLast &&
      convInfo.filterHeight === convInfo.inHeight &&
      convInfo.filterWidth === convInfo.inWidth &&
      convInfo.padInfo.type === 'VALID';
  const useNaiveConv2d = env().getBool('WEBGPU_USE_NAIVE_CONV2D_DEBUG');

  if (!useNaiveConv2d &&
      (sameSize ||
       (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
        convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
        convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
        (convInfo.padInfo.type === 'SAME' ||
         convInfo.padInfo.type === 'VALID')))) {
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

  const thresholdFlagValue =
      env().getNumber('WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL');
  const thresholdToIncreaseWorkgroups = thresholdFlagValue > -1 ?
      thresholdFlagValue :
      backend.thresholdToIncreaseWorkgroups;
  const workgroupsBy32x32 = convInfo.batchSize *
      Math.ceil((convInfo.outHeight * convInfo.outWidth) / 32) *
      Math.ceil(convInfo.outChannels / 32);
  if (env().getBool('WEBGPU_CONV_SEPARATE_IM2COL_SHADER') ||
      workgroupsBy32x32 <= thresholdToIncreaseWorkgroups) {
    return conv2dWithIm2Col({
      x,
      filter,
      convInfo,
      backend,
      bias,
      preluActivationWeights,
      leakyreluAlpha,
      activation
    });
  }

  let program: WebGPUProgram;
  const padInfo = [convInfo.padInfo.top, convInfo.padInfo.left];
  const dimensions = [
    {type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth]},
    {type: 'int32', data: [...padInfo]},
    {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]},
    {type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth]}
  ];
  if (useNaiveConv2d) {
    program = new Conv2DNaiveProgram(
        convInfo, hasBias, activation, hasPreluActivationWeights);
  } else {
    const dimAOuter = isChannelsLast ? convInfo.outHeight * convInfo.outWidth :
                                       convInfo.outChannels;
    const dimBOuter = isChannelsLast ? convInfo.outChannels :
                                       convInfo.outHeight * convInfo.outWidth;
    const dimInner =
        convInfo.filterHeight * convInfo.filterWidth * convInfo.inChannels;
    dimensions.push(
        {type: 'int32', data: [dimAOuter]}, {type: 'int32', data: [dimBOuter]},
        {type: 'int32', data: [dimInner]});

    // Experiments show that sequential access is more friendly for Intel GPUs.
    const sequentialAccessByThreads = backend.adapterInfo.isIntel();
    program = new Conv2DMMProgram(
        convInfo, dimAOuter, dimBOuter, dimInner, hasBias, activation,
        hasPreluActivationWeights, sequentialAccessByThreads);
  }

  const intermediates: TensorInfo[] = [];
  const inputVar: TensorInfo[] = [x, filter];
  if (hasBias) {
    if (!isChannelsLast && bias.shape.length === 1) {
      bias = reshape(
          {inputs: {x: bias}, backend, attrs: {shape: [bias.shape[0], 1, 1]}});
      intermediates.push(bias);
    }
    inputVar.push(bias);
  }
  if (hasPreluActivationWeights) {
    if (!isChannelsLast && preluActivationWeights.shape.length === 1) {
      preluActivationWeights = reshape({
        inputs: {x: preluActivationWeights},
        backend,
        attrs: {shape: [preluActivationWeights.shape[0], 1, 1]}
      });
      intermediates.push(preluActivationWeights);
    }
    inputVar.push(preluActivationWeights);
  }
  if (activation === 'leakyrelu') {
    dimensions.push({type: 'float32', data: [leakyreluAlpha]});
    program.uniforms += ' alpha : f32,';
  }
  const out = backend.runWebGPUProgram(program, inputVar, x.dtype, dimensions);
  for (const i of intermediates) {
    backend.disposeData(i.dataId);
  }
  return out;
}
