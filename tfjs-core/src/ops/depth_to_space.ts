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

import {ENGINE} from '../engine';
import {DepthToSpace, DepthToSpaceAttrs, DepthToSpaceInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike4D} from '../types';
import * as util from '../util';

import {op} from './operation';

/**
 * Rearranges data from depth into blocks of spatial data. More specifically,
 * this op outputs a copy of the input tensor where values from the `depth`
 * dimension are moved in spatial blocks to the `height` and `width` dimensions.
 * The attr `blockSize` indicates the input block size and how the data is
 * moved.
 *
 *  - Chunks of data of size `blockSize * blockSize` from depth are rearranged
 * into non-overlapping blocks of size `blockSize x blockSize`
 *
 *  - The width the output tensor is `inputWidth * blockSize`, whereas the
 * height is `inputHeight * blockSize`
 *
 *  - The Y, X coordinates within each block of the output image are determined
 * by the high order component of the input channel index
 *
 *  - The depth of the input tensor must be divisible by `blockSize *
 * blockSize`
 *
 * The `dataFormat` attr specifies the layout of the input and output tensors
 * with the following options: "NHWC": [ `batch, height, width, channels` ]
 * "NCHW": [ `batch, channels, height, width` ]
 *
 * ```js
 * const x = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
 * const blockSize = 2;
 * const dataFormat = "NHWC";
 *
 * tf.depthToSpace(x, blockSize, dataFormat).print();
 * ```
 *
 * @param x The input tensor of rank 4
 * @param blockSIze  An `int` that is `>= 2`. The size of the spatial block
 * @param dataFormat An optional string from: "NHWC", "NCHW". Defaults to "NHWC"
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function depthToSpace_(
    x: Tensor4D|TensorLike4D, blockSize: number,
    dataFormat: 'NHWC'|'NCHW' = 'NHWC'): Tensor4D {
  const $x = convertToTensor(x, 'x', 'depthToSpace') as Tensor4D;

  const inputHeight = (dataFormat === 'NHWC') ? $x.shape[1] : $x.shape[2];
  const inputWidth = (dataFormat === 'NHWC') ? $x.shape[2] : $x.shape[3];
  const inputDepth = (dataFormat === 'NHWC') ? $x.shape[3] : $x.shape[1];

  util.assert(
      inputHeight * blockSize >= 0,
      () => `Negative dimension size caused by overflow when multiplying
    ${inputHeight} and ${blockSize}  for depthToSpace with input shape
    ${$x.shape}`);

  util.assert(
      inputWidth * blockSize >= 0,
      () => `Negative dimension size caused by overflow when multiplying
    ${inputWidth} and ${blockSize} for depthToSpace with input shape
        ${$x.shape}`);

  util.assert(
      (inputDepth % (blockSize * blockSize) === 0),
      () => `Dimension size must be evenly divisible by ${
          blockSize * blockSize} but is ${
          inputDepth} for depthToSpace with input shape ${$x.shape}`);

  const inputs: DepthToSpaceInputs = {x: $x};
  const attrs: DepthToSpaceAttrs = {blockSize, dataFormat};

  return ENGINE.runKernel(
      DepthToSpace, inputs as {} as NamedTensorMap,
      attrs as {} as NamedAttrMap);
}

export const depthToSpace = op({depthToSpace_});
