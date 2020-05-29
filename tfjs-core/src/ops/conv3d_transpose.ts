/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import {Tensor4D, Tensor5D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {conv3DBackpropInput} from './conv3d_backprop_input';
import {op} from './operation';

/**
 * Computes the transposed 3D convolution of a volume, also known as a
 * deconvolution.
 *
 * @param x The input image, of rank 5 or rank 4, of shape
 *   `[batch, depth, height, width, inDepth]`. If rank 4, batch of 1 is assumed.
 * @param filter The filter, rank 4, of shape
 *     `[depth, filterHeight, filterWidth, outDepth, inDepth]`.
 *     `inDepth` must match `inDepth` in `x`.
 * @param outputShape Output shape, of rank 5 or rank 4:
 *     `[batch, depth, height, width, outDepth]`. If rank 3, batch of 1 is
 *    assumed.
 * @param strides The strides of the original convolution:
 *     `[strideDepth, strideHeight, strideWidth]`.
 * @param pad  The type of padding algorithm used in the non-transpose version
 *    of the op.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function conv3dTranspose_<T extends Tensor4D|Tensor5D>(
    x: T|TensorLike, filter: Tensor5D|TensorLike,
    outputShape:
        [number, number, number, number,
         number]|[number, number, number, number],
    strides: [number, number, number]|number, pad: 'valid'|'same'): T {
  const $x = convertToTensor(x, 'x', 'conv3dTranspose');
  const $filter = convertToTensor(filter, 'filter', 'conv3dTranspose');

  return conv3DBackpropInput(outputShape, $x, $filter, strides, pad);
}

export const conv3dTranspose = op({conv3dTranspose_});
