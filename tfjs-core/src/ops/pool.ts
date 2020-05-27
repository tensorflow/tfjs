/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {Tensor3D, Tensor4D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {reshape} from './array_ops';
import {avgPool} from './avg_pool';
import {batchToSpaceND} from './batch_to_space_nd';
import * as conv_util from './conv_util';
import {maxPool} from './max_pool';
import {op} from './operation';
import {spaceToBatchND} from './space_to_batch_nd';

/**
 * Performs an N-D pooling operation
 *
 * @param input The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param windowShape The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param poolingType The type of pooling, either 'max' or 'avg'.
 * @param pad The type of padding algorithm:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *         https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in dilated pooling. Defaults to `[1, 1]`. If `dilationRate` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function pool_<T extends Tensor3D|Tensor4D>(
    input: T|TensorLike, windowShape: [number, number]|number,
    poolingType: 'avg'|'max', pad: 'valid'|'same'|number,
    dilations?: [number, number]|number, strides?: [number, number]|number) {
  if (dilations == null) {
    dilations = [1, 1];
  }
  if (strides == null) {
    strides = 1;
  }
  if (pad === 0) {
    pad = 'valid';
  }

  const $x = convertToTensor(input, 'x', 'maxPool');
  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;

  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }

  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in pool: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);

  const convInfo = conv_util.computePool2DInfo(
      x4D.shape, windowShape, strides, dilations, pad);
  const dilation: [number, number] =
      [convInfo.dilationHeight, convInfo.dilationWidth];

  // The following implementation does batchToSpace(pool(spaceToBatch(x)))
  // whenever dilation > 1 since the TF kernels do not support dilation > 1.
  // tslint:disable-next-line:max-line-length
  // https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/nn_ops.py#L1037

  let basePadding: number[][];
  if (pad === 'same') {
    basePadding = withSpaceToBatchBasePaddings(
        [convInfo.filterHeight, convInfo.filterWidth], dilation);
  } else {
    basePadding = [[0, 0], [0, 0]];
  }

  const isDilationOne = dilation[0] === 1 && dilation[1] === 1;
  const [adjustedPadding, adjustedCrops] = requiredSpaceToBatchPaddings(
      [convInfo.inHeight, convInfo.inWidth], dilation, basePadding);
  const convertedPad = isDilationOne ? pad : 'valid';
  const convertedX =
      isDilationOne ? x4D : spaceToBatchND(x4D, dilation, adjustedPadding);

  const forwardOp = poolingType === 'avg' ?
      () => avgPool(convertedX, windowShape, strides, convertedPad) :
      () => maxPool(convertedX, windowShape, strides, convertedPad);
  const y = forwardOp();

  const res = isDilationOne ? y : batchToSpaceND(y, dilation, adjustedCrops);

  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
  }

  return res as T;
}

// Helper function to compute crops and paddings for pool with dilation > 1.
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/array_ops.py#L2184
function requiredSpaceToBatchPaddings(
    inputShape: [number, number], blockShape: [number, number],
    basePadding: number[][]) {
  const padStart = basePadding.map(b => b[0]);
  const origPadEnd = basePadding.map(b => b[1]);
  const fullInputShape = inputShape.concat(padStart, origPadEnd);
  const padEndExtra = blockShape.map((b, i) => (b - fullInputShape[i] % b) % b);
  const padEnd = origPadEnd.map((s, i) => s + padEndExtra[i]);
  const paddings = blockShape.map((_, i) => [padStart[i], padEnd[i]]);
  const crops = blockShape.map((_, i) => [0, padEndExtra[i]]);
  return [paddings, crops];
}

// Helper function to compute base paddings for pool with dilation > 1.
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/nn_ops.py#L524
function withSpaceToBatchBasePaddings(
    filterShape: [number, number], dilation: [number, number]) {
  // Spatial dimensions of the filters and the upsampled filters in which we
  // introduce (rate - 1) zeros between consecutive filter values.
  const dilatedFilterShape = filterShape.map((s, i) => {
    return s + (s - 1) * (dilation[i] - 1);
  });
  const padExtraShape = dilatedFilterShape.map(s => s - 1);

  // When padding is odd, we pad more at end, following the same
  // convention as conv2d.
  const padExtraStart = padExtraShape.map(s => Math.floor(s / 2));
  const padExtraEnd = padExtraShape.map((s, i) => s - padExtraStart[i]);
  return padExtraShape.map((_, i) => {
    return [padExtraStart[i], padExtraEnd[i]];
  });
}

export const pool = op({pool_});
