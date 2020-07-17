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

import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {parseAxisParam} from '../util';

import {expandShapeToKeepDim} from './axis_util';
import {cast} from './cast';
import {mean} from './mean';
import {op} from './operation';
import {reshape} from './reshape';
import {square} from './square';
import {sub} from './sub';

/**
 * Calculates the mean and variance of `x`. The mean and variance are
 * calculated by aggregating the contents of `x` across `axes`. If `x` is
 * 1-D and `axes = [0]` this is just the mean and variance of a vector.
 *
 * @param x The input tensor.
 * @param axis The dimension(s) along with to compute mean and
 *     variance. By default it reduces all dimensions.
 * @param keepDims If true, the moments have the same dimensionality as the
 *     input.
 * @return An object with two keys: `mean` and `variance`.
 */
/** @doc {heading: 'Operations', subheading: 'Normalization'} */
function moments_(
    x: Tensor|TensorLike, axis: number|number[] = null,
    keepDims = false): {mean: Tensor, variance: Tensor} {
  x = convertToTensor(x, 'x', 'moments');
  const axes = parseAxisParam(axis, x.shape);
  const xMean = mean(x, axes, keepDims);
  let keepDimsShape = xMean.shape;
  if (!keepDims) {
    keepDimsShape = expandShapeToKeepDim(xMean.shape, axes);
  }
  const devSquared =
      square(sub(cast(x, 'float32'), reshape(xMean, keepDimsShape)));
  const variance = mean(devSquared, axes, keepDims);
  return {mean: xMean, variance};
}

export const moments = op({moments_});
