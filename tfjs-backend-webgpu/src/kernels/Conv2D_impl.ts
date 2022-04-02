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
import {Conv2DMMVec4Program} from '../conv2d_mm_vec4_webgpu';
import {Conv2DMMProgram} from '../conv2d_mm_webgpu';
import {Conv2DNaiveProgram} from '../conv2d_naive_webgpu';
import {Im2ColProgram} from '../im2col_webgpu';
import {MatMulPackedProgram} from '../matmul_packed_webgpu';

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
  const xShape = x.shape;
  const isChannelsLast = convInfo.dataFormat === 'channelsLast';
  const transposeA = false;
  const transposeB = false;

  const sameSize = convInfo.filterHeight === convInfo.inHeight &&
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
    const targetShape = isChannelsLast ? xShape[0] * xShape[1] * xShape[2] :
                                         xShape[0] * xShape[2] * xShape[3];
    xReshaped = reshape({
      inputs: {x},
      backend,
      attrs: {shape: [1, targetShape, convInfo.inChannels]}
    });
    filterReshaped = reshape({
      inputs: {x: filter},
      backend,
      attrs: {shape: [1, convInfo.inChannels, convInfo.outChannels]}
    });
  }

  const result = batchMatMulImpl({
    a: xReshaped,
    b: filterReshaped,
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

// Implements the im2row algorithm as outlined in "High Performance
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
  // column of a new matrix with shape [filterWidth * filterHeight *
  // inChannels, outHeight * outWidth]. The filter is also rearranged so each
  // output channel forms a row of a new matrix with shape [outChannels,
  // filterWidth * filterHeight * inChannels]. The convolution is then
  // computed by multiplying these matrices and reshaping the result.
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
  const x2ColShape = [numCols, sharedDim];
  const transposeA = false;
  const transposeB = false;

  const intermediates: TensorInfo[] = [];

  const xSqueezed =
      reshape({inputs: {x}, backend, attrs: {shape: x.shape.slice(1)}});
  const w2Row = reshape(
      {inputs: {x: filter}, backend, attrs: {shape: [1, sharedDim, -1]}});

  intermediates.push(xSqueezed);
  intermediates.push(w2Row);

  const im2ColProgram = new Im2ColProgram(x2ColShape, isChannelsLast);
  const dimensions = [
    {type: 'int32', data: [padInfo.left, padInfo.top]},      // Padding.
    {type: 'int32', data: [strideWidth, strideHeight]},      // Stride.
    {type: 'int32', data: [dilationWidth, dilationHeight]},  // Dilation.
    {type: 'int32', data: [outWidth]},
    {type: 'int32', data: [inChannels * filterWidth]},  // itemsPerBlockRow.
    {type: 'int32', data: [inChannels]}
  ];
  const im2Col = backend.runWebGPUProgram(
      im2ColProgram, [xSqueezed], xSqueezed.dtype, dimensions);
  const im2Col3D = reshape({
    inputs: {x: im2Col},
    backend,
    attrs: {shape: [1, x2ColShape[0], x2ColShape[1]]}
  });
  intermediates.push(im2Col);
  intermediates.push(im2Col3D);
  const a3dShape: [number, number, number] = [1, x2ColShape[0], x2ColShape[1]];
  const matMulProgram = new MatMulPackedProgram(
      a3dShape, [1, numCols, convInfo.outChannels],
      env().get('WEBGPU_MATMUL_WORK_PER_THREAD') as number, transposeA,
      transposeB, bias, activation, preluActivationWeights);
  const dimAOuter = a3dShape[1];
  const dimInner = a3dShape[2];
  const dimBOuter = convInfo.outChannels;
  const matmulDimensions = [
    {type: 'int32', data: [dimAOuter]}, {type: 'int32', data: [dimBOuter]},
    {type: 'int32', data: [dimInner]}
  ];
  const inputs: TensorInfo[] = [im2Col3D, w2Row];
  if (bias) {
    inputs.push(bias);
  }
  if (preluActivationWeights) {
    inputs.push(preluActivationWeights);
  }
  if (activation === 'leakyrelu') {
    dimensions.push({type: 'float32', data: [leakyreluAlpha]});
    matMulProgram.uniforms += ' alpha : f32,';
  }
  const result: TensorInfo = backend.runWebGPUProgram(
      matMulProgram, inputs, im2Col3D.dtype, matmulDimensions);

  const outShape = isChannelsLast ?
      [1, outHeight, outWidth, convInfo.outChannels] :
      [1, convInfo.outChannels, outHeight, outWidth];
  const out = reshape({inputs: {x: result}, backend, attrs: {shape: outShape}});

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

  let program: Conv2DMMProgram|Conv2DNaiveProgram|Conv2DMMVec4Program;
  const sameSize = convInfo.filterHeight === convInfo.inHeight &&
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

  if (env().getBool('WEBGPU_CONV_SEPARATE_IM2COL_SHADER') && x.shape[0] === 1) {
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
  const useNaive = env().getBool('WEBGPU_USE_NAIVE_CONV2D');

  const useVec4 =
      (convInfo.inChannels % 4 === 0 ||
       (convInfo.inChannels === 3 && convInfo.padInfo.type === 'VALID')) &&
      convInfo.outChannels % 4 === 0;

  const padInfo = [convInfo.padInfo.top, convInfo.padInfo.left];
  const dimensions = [
    {type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth]},
    {type: 'int32', data: [...padInfo]},
    {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]},
    {type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth]}
  ];
  if (useNaive) {
    // TODO(kainino0x): This may be obsolete, but is kept for reference.
    program = new Conv2DNaiveProgram(
        convInfo, hasBias, activation, hasPreluActivationWeights);
  } else {
    if (useVec4) {
      program = new Conv2DMMVec4Program(
          convInfo, hasBias, activation, hasPreluActivationWeights);
    } else {
      program = new Conv2DMMProgram(
          convInfo, hasBias, activation, hasPreluActivationWeights);
    }
    const dimAOuter = convInfo.outShape[1] * convInfo.outShape[2];
    const dimBOuter = convInfo.outShape[3];
    const dimInner =
        convInfo.filterHeight * convInfo.filterWidth * convInfo.inShape[3];
    dimensions.push(
        {type: 'int32', data: [dimAOuter]}, {type: 'int32', data: [dimBOuter]},
        {type: 'int32', data: [dimInner]});
  }

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
