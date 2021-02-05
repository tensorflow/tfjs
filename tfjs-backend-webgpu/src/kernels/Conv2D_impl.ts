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

import {batchMatMulImpl} from './BatchMatMul_impl';
import {Im2ColProgram} from './im2col_webgpu';
import {MatMulPackedProgram} from './matmul_packed_webgpu';
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
export function conv2dByMatMul({
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

  const targetShape = isChannelsLast ? xShape[0] * xShape[1] * xShape[2] :
                                       xShape[0] * xShape[2] * xShape[3];
  const xReshaped = reshape({
    inputs: {x},
    backend,
    attrs: {shape: [1, targetShape, convInfo.inChannels]}
  });
  const filterReshaped = reshape({
    inputs: {x: filter},
    backend,
    attrs: {shape: [1, convInfo.inChannels, convInfo.outChannels]}
  });

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
export function conv2dWithIm2Col({
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
    outWidth,
    outHeight,
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

  const im2ColProgram =
      new Im2ColProgram(x2ColShape, xSqueezed.shape, convInfo);
  const im2Col =
      backend.runWebGPUProgram(im2ColProgram, [xSqueezed], xSqueezed.dtype);
  const im2Col3D = reshape({
    inputs: {x: im2Col},
    backend,
    attrs: {shape: [1, x2ColShape[0], x2ColShape[1]]}
  });
  intermediates.push(im2Col);
  intermediates.push(im2Col3D);

  const matMulProgram = new MatMulPackedProgram(
      [1, x2ColShape[0], x2ColShape[1]], [1, numCols, convInfo.outChannels],
      env().get('WEBGPU_MATMUL_WORK_PER_THREAD') as number, transposeA,
      transposeB);
  const result: TensorInfo = backend.runWebGPUProgram(
      matMulProgram, [im2Col3D, w2Row], im2Col3D.dtype);

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
