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
import {customGrad} from '../gradients';
import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {parseAxisParam, sizeFromShape} from '../util';

import {computeOutAndReduceShapes} from './axis_util';
import {cast} from './cast';
import {div} from './div';
import {mul} from './mul';
import {ones} from './ones';
import {op} from './operation';
import {reshape} from './reshape';
import {scalar} from './scalar';
import {sum} from './sum';

/**
 * Computes the mean of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces `x` along the dimensions given in `axis`. Unless `keepDims` is
 * true, the rank of the `tf.Tensor` is reduced by 1 for each entry in `axis`.
 * If `keepDims` is true, the reduced dimensions are retained with length 1.
 * If `axis` has no entries, all dimensions are reduced, and a `tf.Tensor` with
 * a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.mean().print();  // or tf.mean(a)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.mean(axis).print();  // or tf.mean(x, axis)
 * ```
 *
 * @param x The input tensor.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 */
/** @doc {heading: 'Operations', subheading: 'Reduction'} */
function mean_<T extends Tensor>(
    x: Tensor|TensorLike, axis: number|number[] = null, keepDims = false): T {
  const $x = convertToTensor(x, 'x', 'mean');

  const axes = parseAxisParam(axis, $x.shape);
  const shapes = computeOutAndReduceShapes($x.shape, axes);
  const reduceShape = shapes[1];
  const reduceSize = sizeFromShape(reduceShape);

  // Use a custom gradient to bypass 2 gradient backprops since mean is used
  // extremely often.
  const customOp = customGrad((x: Tensor) => {
    const reduceSizeScalar = scalar(reduceSize);
    // Cast if needed.
    const xReduce = reduceSizeScalar.dtype === x.dtype ?
        x :
        cast(x, reduceSizeScalar.dtype);
    const res = div(xReduce, reduceSizeScalar);
    const value = sum(res, axis, keepDims);

    const gradFunc = (dy: Tensor) => {
      const expandedDyShape = x.shape.slice();
      axes.forEach(axis => {
        expandedDyShape[axis] = 1;
      });
      const expandedDy = reshape(dy, expandedDyShape);
      const derX = div(mul(expandedDy, ones(x.shape, 'float32')), reduceSize);
      return derX;
    };
    return {value, gradFunc};
  });

  return customOp($x) as T;
}

export const mean = op({mean_});
