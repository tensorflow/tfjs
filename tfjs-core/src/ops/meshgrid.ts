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

import {matMul} from './mat_mul';
import {ones} from './ones';
import {reshape} from './reshape';
import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {sizeFromShape} from '../util_base';

export function meshgrid<T extends Tensor>(
    x?: T|TensorLike, y?: T|TensorLike, indexing = 'xy'): T[] {
  if (indexing !== 'xy' && indexing !== 'ij') {
    throw new TypeError(
        `${indexing} is not a valid third argument to meshgrid`);
  }
  if (!x) {
    return [];
  }
  let $x = convertToTensor(
      x, 'x', 'meshgrid', x instanceof Tensor ? x.dtype : 'float32');
  if ($x.rank === 0) {
    throw new TypeError('meshgrid expects a tensor with rank >= 1');
  }

  if (!y) {
    return [$x];
  }
  let $y = convertToTensor(
      y, 'y', 'meshgrid', y instanceof Tensor ? y.dtype : 'float32');
  if ($y.rank === 0) {
    throw new TypeError('meshgrid expects a tensor with rank >= 1');
  }

  const dtype = $x.dtype;
  const w = sizeFromShape($x.shape);
  const h = sizeFromShape($y.shape);

  if (indexing === 'xy') {
    $x = reshape($x, [1, -1]) as T;
    $y = reshape($y, [-1, 1]) as T;
    return [matMul(ones([h, 1], dtype), $x), matMul($y, ones([1, w], dtype))];
  }

  $x = reshape($x, [-1, 1]) as T;
  $y = reshape($y, [1, -1]) as T;
  return [matMul($x, ones([1, h], dtype)), matMul(ones([w, 1], dtype), $y)];
}
