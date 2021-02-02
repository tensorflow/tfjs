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
import {AvgPool3D, AvgPool3DAttrs, AvgPool3DInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor4D, Tensor5D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {cast} from './cast';
import {op} from './operation';
import {reshape} from './reshape';

/**
 * Computes the 3D average pooling.
 *
 * ```js
 * const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
 * const result = tf.avgPool3d(x, 2, 1, 'valid');
 * result.print();
 * ```
 *
 * @param x The input tensor, of rank 5 or rank 4 of shape
 *     `[batch, depth, height, width, inChannels]`.
 * @param filterSize The filter size:
 *     `[filterDepth, filterHeight, filterWidth]`.
 *     If `filterSize` is a single number,
 *     then `filterDepth == filterHeight == filterWidth`.
 * @param strides The strides of the pooling:
 *     `[strideDepth, strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideDepth == strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function avgPool3d_<T extends Tensor4D|Tensor5D>(
    x: T|TensorLike, filterSize: [number, number, number]|number,
    strides: [number, number, number]|number, pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil',
    dataFormat: 'NDHWC'|'NCDHW' = 'NDHWC'): T {
  const $x = convertToTensor(x, 'x', 'avgPool3d', 'float32');

  let x5D = $x as Tensor5D;
  let reshapedTo5D = false;
  if ($x.rank === 4) {
    reshapedTo5D = true;
    x5D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]]);
  }

  util.assert(
      x5D.rank === 5,
      () => `Error in avgPool3d: x must be rank 5 but got rank ${x5D.rank}.`);
  util.assert(
      dataFormat === 'NDHWC',
      () => `Error in avgPool3d: Only NDHWC is currently supported, ` +
          `but got dataFormat of ${dataFormat}`);

  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in avgPool3d: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const inputs: AvgPool3DInputs = {x: x5D};

  const attrs:
      AvgPool3DAttrs = {filterSize, strides, pad, dimRoundingMode, dataFormat};

  // tslint:disable-next-line: no-unnecessary-type-assertion
  let res = ENGINE.runKernel(
                AvgPool3D, inputs as {} as NamedTensorMap,
                attrs as {} as NamedAttrMap) as T;

  res = cast(res, x5D.dtype);

  if (reshapedTo5D) {
    return reshape(
               res, [res.shape[1], res.shape[2], res.shape[3], res.shape[4]]) as
        T;
  }

  return res;
}

export const avgPool3d = op({avgPool3d_});
