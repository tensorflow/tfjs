/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {gather} from './gather';
import {reshape} from './reshape';
import {squeeze} from './squeeze';
import {whereAsync} from './where_async';

/**
 * Apply boolean mask to tensor.
 *
 * ```js
 * const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
 * const mask = tf.tensor1d([1, 0, 1], 'bool');
 * const result = await tf.booleanMaskAsync(tensor, mask);
 * result.print();
 * ```
 *
 * @param tensor N-D tensor.
 * @param mask K-D boolean tensor, K <= N and K must be known statically.
 * @param axis A 0-D int Tensor representing the axis in tensor to mask from.
 *     By default, axis is 0 which will mask from the first dimension.
 *     Otherwise K + axis <= N.
 */
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
async function booleanMaskAsync_(
    tensor: Tensor|TensorLike, mask: Tensor|TensorLike,
    axis?: number): Promise<Tensor> {
  const $tensor = convertToTensor(tensor, 'tensor', 'boolMask');
  const $mask = convertToTensor(mask, 'mask', 'boolMask', 'bool');

  const axisFrom = axis == null ? 0 : axis;
  const maskDim = $mask.rank;
  const tensorShape = $tensor.shape;

  util.assert(maskDim > 0, () => 'mask cannot be scalar');
  util.assertShapesMatch(
      tensorShape.slice(axisFrom, axisFrom + maskDim), $mask.shape,
      `mask's shape must match the first K dimensions of tensor's shape,`);

  let leadingSize = 1;
  for (let i = axisFrom; i < axisFrom + maskDim; i++) {
    leadingSize *= tensorShape[i];
  }
  const targetTensorShape =
      tensorShape.slice(0, axisFrom)
          .concat([leadingSize], tensorShape.slice(axisFrom + maskDim));
  const reshapedTensor = reshape($tensor, targetTensorShape);
  const reshapedMask = reshape($mask, [-1]);
  const positivePositions = await whereAsync(reshapedMask);
  const indices = squeeze(positivePositions, [1]);

  const res = gather(reshapedTensor, indices, axisFrom);

  // Ensure no memory leak.
  if (tensor !== $tensor) {
    $tensor.dispose();
  }
  if (mask !== $mask) {
    $mask.dispose();
  }
  indices.dispose();
  reshapedTensor.dispose();
  reshapedMask.dispose();
  positivePositions.dispose();

  return res;
}

export const booleanMaskAsync = booleanMaskAsync_;
