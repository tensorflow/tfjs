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

import {backend_util, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {assertAndGetBroadcastShape} from
// '../../../tfjs-core/src/ops/broadcast_util';
import {MathBackendWebGL} from '../backend_webgl';
import {Im2ColPackedProgram} from '../im2col_packed_gpu';
import {mapActivationToShaderProgram} from '../kernel_utils/kernel_funcs_utils';
import {MatMulPackedProgram} from '../mulmat_packed_gpu';
import * as webgl_util from '../webgl_util';

import {batchMatMulImpl, batchMatMulMrt2x2Impl, MATMUL_SHARED_DIM_THRESHOLD} from './BatchMatMul_impl';
import {identity} from './Identity';
import {reshape} from './Reshape';

type Conv2DConfig = {
  x: TensorInfo,
  filter: TensorInfo,
  convInfo: backend_util.Conv2DInfo,
  backend: MathBackendWebGL,
  bias?: TensorInfo,
  preluActivationWeights?: TensorInfo,
  leakyreluAlpha?: number,
  activation?: backend_util.Activation
};

// Both conv2dByMatMul and conv2dWithIm2Row fuse height and width into one
// dimension to compute batchMatMul, so bias and activation weights are also
// supposed to fuse the two dimensions into one.
//
// This function computes the target shape for fusing height and width
// dimensions. Returning null means the shape is already compatible.
//
// Even though the bias is not supposed to be a 3-D or a 4-D (including
// batch) tensor and PReLU activiation weights is not supposed to be a 4-D
// tensor, we still need to support them, because we haven't disabled
// them for NHWC format.
// https://github.com/tensorflow/tfjs/blob/b53bd47e880367ae57493f0ea628abaf08db2d5d/tfjs-core/src/ops/fused/conv2d.ts#L181-L196
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
export function conv2dByMatMulMrt2x2({
  x,
  filter,
  convInfo,
  backend,
  bias = null,
  preluActivationWeights = null,
  leakyreluAlpha = 0,
  activation = null
}: Conv2DConfig) {
  // Reshapes conv2D input to 2D tensors, uses matMul and then reshape the
  // result from 2D to 4D.

  webgl_util.assert(
      convInfo.batchSize === 1, 'MatMul MRT does not support batch!');
  webgl_util.assert(bias == null, 'MatMul MRT does not support bias!');
  webgl_util.assert(
      leakyreluAlpha == null, 'MatMul MRT does not support leakyreluAlpha!');
  webgl_util.assert(
      preluActivationWeights == null,
      'MatMul MRT does not support preluActivationWeights!');
  const isChannelsLast = convInfo.dataFormat === 'channelsLast';
  webgl_util.assert(isChannelsLast, 'MatMul MRT does not support NCHW!');

  let out: TensorInfo;
  const intermediates: TensorInfo[] = [];

  const numCols = convInfo.outHeight * convInfo.outWidth;
  const xReshaped = reshape({
    inputs: {x},
    backend,
    attrs: {
      shape: isChannelsLast ?
          [convInfo.batchSize, numCols, convInfo.inChannels] :
          [convInfo.batchSize, convInfo.inChannels, numCols]
    }
  });
  const filterReshaped = reshape({
    inputs: {x: filter},
    backend,
    attrs: {shape: [1, convInfo.inChannels, convInfo.outChannels]}
  });
  const result = batchMatMulMrt2x2Impl({
    a: isChannelsLast ? xReshaped : filterReshaped,
    b: isChannelsLast ? filterReshaped : xReshaped,
    transposeA: !isChannelsLast,
    transposeB: false,
    backend,
    bias,
    activation,
    preluActivationWeights,
    leakyreluAlpha
  });

  out = reshape(
      {inputs: {x: result}, backend, attrs: {shape: convInfo.outShape}});

  intermediates.push(xReshaped);
  intermediates.push(filterReshaped);
  intermediates.push(result);

  for (const i of intermediates) {
    backend.disposeIntermediateTensorInfo(i);
  }

  return out;
}


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
  // Reshapes conv2D input to 2D tensors, uses matMul and then reshape the
  // result from 2D to 4D.
  const xShape = x.shape;
  const xTexData = backend.texData.get(x.dataId);
  const sharedMatMulDim = convInfo.inChannels;
  const outerShapeX = xShape[0] * xShape[1] * xShape[2];
  const outerShapeFilter = convInfo.outChannels;
  const isChannelsLast = convInfo.dataFormat === 'channelsLast';
  const transposeA = false;
  const transposeB = false;

  let out: TensorInfo;
  const intermediates: TensorInfo[] = [];

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

  // TODO: Once reduction ops are packed, batchMatMul will always be packed
  // and we can remove this condition.
  const batchMatMulWillBeUnpacked =
      (outerShapeX === 1 || outerShapeFilter === 1) &&
      sharedMatMulDim > MATMUL_SHARED_DIM_THRESHOLD;

  // The algorithm in the if condition assumes (1) the output will be packed,
  // (2) x is packed, (3) x isChannelsLast, (4)  x's packed texture is already
  // on GPU, (5) col is odd, (6) the width, height and inChannels are the same
  // for xTexData.shape and xShape.
  const canOptimize = !batchMatMulWillBeUnpacked && xTexData.isPacked &&
      isChannelsLast && xTexData.texture != null && xShape[2] % 2 !== 0 &&
      util.arraysEqual(xTexData.shape.slice(-3), xShape.slice(-3));

  if (canOptimize) {
    // We avoid expensive packed 2x2 reshape by padding col count to next,
    // even number. When col is odd, the result of packed batchMatMul is
    // the same (has the same texture layout and and values in the texture) as
    // it is for next even col. We make the odd-cols tensor to look like
    // even-cols tensor before the operation and, after the batchMatMul,
    // fix the even-cols result to have odd number of cols.
    const targetShape = xShape[0] * xShape[1] * (xShape[2] + 1);
    const xReshaped: TensorInfo = {
      dataId: x.dataId,
      shape: [1, targetShape, convInfo.inChannels],
      dtype: x.dtype
    };
    // xTexData.shape gets referenced from GPGPUBinary.inShapeInfos.
    // Decrementing col count, after batchMatMul->...->compileProgram leads to
    // invalid col count within the reference in GPGPUBinary.inShapeInfos.
    // Alternative fix would be to provide a copy to GPGPUBinary.inShapeInfos
    // in compileProgram method, but that would affect compilation of all
    // programs - instead, provide a copy here, with even col count, before
    // calling batchMatMul->...->compileProgram and after that, the original
    // xTexData.shape is restored.
    const originalXTexDataShape = xTexData.shape;
    xTexData.shape = xTexData.shape.slice();
    xTexData.shape[xTexData.shape.length - 2]++;
    util.assert(
        webgl_util.isReshapeFree(xTexData.shape, xReshaped.shape),
        () => `packed reshape ${xTexData.shape} to ${
            xReshaped.shape} isn't free`);
    const filterReshaped = reshape({
      inputs: {x: filter},
      backend,
      attrs: {shape: [1, convInfo.inChannels, convInfo.outChannels]}
    });
    intermediates.push(filterReshaped);
    const pointwiseConv = batchMatMulImpl({
      a: xReshaped,
      b: filterReshaped,
      backend,
      transposeA,
      transposeB,
      bias,
      activation,
      preluActivationWeights,
      leakyreluAlpha
    });

    const pointwiseConvTexData = backend.texData.get(pointwiseConv.dataId);
    util.assert(
        pointwiseConvTexData.isPacked,
        () => 'batchMatMul result is expected to be packed');
    // Restore the input shape to original.
    xTexData.shape = originalXTexDataShape;
    // Set the output shape - there is no need for expensive reshape as data
    // layout is already correct.
    pointwiseConvTexData.shape = convInfo.outShape;

    out = identity({inputs: {x: pointwiseConv}, backend});
    out.shape = convInfo.outShape;

    intermediates.push(pointwiseConv);
  } else {
    const numCols = convInfo.outHeight * convInfo.outWidth;
    const xReshaped = reshape({
      inputs: {x},
      backend,
      attrs: {
        shape: isChannelsLast ?
            [convInfo.batchSize, numCols, convInfo.inChannels] :
            [convInfo.batchSize, convInfo.inChannels, numCols]
      }
    });
    const filterReshaped = reshape({
      inputs: {x: filter},
      backend,
      attrs: {shape: [1, convInfo.inChannels, convInfo.outChannels]}
    });
    const result = batchMatMulImpl({
      a: isChannelsLast ? xReshaped : filterReshaped,
      b: isChannelsLast ? filterReshaped : xReshaped,
      transposeA: !isChannelsLast,
      transposeB,
      backend,
      bias,
      activation,
      preluActivationWeights,
      leakyreluAlpha
    });

    out = reshape(
        {inputs: {x: result}, backend, attrs: {shape: convInfo.outShape}});

    intermediates.push(xReshaped);
    intermediates.push(filterReshaped);
    intermediates.push(result);
  }

  for (const i of intermediates) {
    backend.disposeIntermediateTensorInfo(i);
  }

  return out;
}

// Implements the im2row algorithm as outlined in "High Performance
// Convolutional Neural Networks for Document Processing" (Suvisoft, 2006)
export function conv2dWithIm2Row({
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
  const x2ColShape = [convInfo.batchSize, sharedDim, numCols];
  const transposeA = true;
  const transposeB = false;

  const intermediates: TensorInfo[] = [];

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

  const w2Row = reshape({
    inputs: {x: filter},
    backend,
    attrs: {shape: [1, sharedDim, util.sizeFromShape(filter.shape) / sharedDim]}
  });
  intermediates.push(w2Row);

  const im2ColProgram = new Im2ColPackedProgram(x2ColShape, convInfo);
  const customValues = [
    x.shape, [convInfo.padInfo.top, convInfo.padInfo.left],
    [convInfo.strideHeight, convInfo.strideWidth],
    [convInfo.dilationHeight, convInfo.dilationWidth], [convInfo.inChannels],
    [convInfo.filterWidth * convInfo.inChannels], [convInfo.outWidth]
  ];
  const im2Col =
      backend.runWebGLProgram(im2ColProgram, [x], 'float32', customValues);
  const im2ColReshaped =
      reshape({inputs: {x: im2Col}, backend, attrs: {shape: x2ColShape}});

  intermediates.push(im2Col);
  intermediates.push(im2ColReshaped);

  const hasBias = bias != null;
  const hasPreluActivationWeights = preluActivationWeights != null;
  const hasLeakyreluAlpha = activation === 'leakyrelu';
  const fusedActivation =
      activation ? mapActivationToShaderProgram(activation, true) : null;
  const matmulProgram = new MatMulPackedProgram(
      isChannelsLast ? im2ColReshaped.shape as [number, number, number] :
                       w2Row.shape as [number, number, number],
      isChannelsLast ? w2Row.shape as [number, number, number] :
                       im2ColReshaped.shape as [number, number, number],
      isChannelsLast ? [convInfo.batchSize, numCols, convInfo.outChannels] :
                       [convInfo.batchSize, convInfo.outChannels, numCols],
      transposeA, transposeB, hasBias, fusedActivation,
      hasPreluActivationWeights, hasLeakyreluAlpha);
  const inputs: TensorInfo[] =
      isChannelsLast ? [im2ColReshaped, w2Row] : [w2Row, im2ColReshaped];
  if (bias) {
    inputs.push(bias);
  }
  if (hasPreluActivationWeights) {
    inputs.push(preluActivationWeights);
  }
  if (hasLeakyreluAlpha) {
    const $leakyreluAlpha = backend.makeTensorInfo(
        [], 'float32',
        util.createScalarValue(leakyreluAlpha as {} as 'float32', 'float32'));
    inputs.push($leakyreluAlpha);
    intermediates.push($leakyreluAlpha);
  }
  const product = backend.runWebGLProgram(matmulProgram, inputs, 'float32');
  const out = reshape(
      {inputs: {x: product}, backend, attrs: {shape: convInfo.outShape}});

  intermediates.push(product);
  for (const i of intermediates) {
    backend.disposeIntermediateTensorInfo(i);
  }

  return out;
}
