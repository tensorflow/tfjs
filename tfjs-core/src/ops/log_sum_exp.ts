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

import {add} from './add';
import {expandShapeToKeepDim} from './axis_util';
import {exp} from './exp';
import {log} from './log';
import {max} from './max';
import {op} from './operation';
import {reshape} from './reshape';
import {sub} from './sub';
import {sum} from './sum';

/**
 * Computes the log(sum(exp(elements across the reduction dimensions)).
 *
 * Reduces the input along the dimensions given in `axis`. Unless `keepDims`
 * is true, the rank of the array is reduced by 1 for each entry in `axis`.
 * If `keepDims` is true, the reduced dimensions are retained with length 1.
 * If `axis` has no entries, all dimensions are reduced, and an array with a
 * single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.logSumExp().print();  // or tf.logSumExp(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.logSumExp(axis).print();  // or tf.logSumExp(a, axis)
 * ```
 * @param x The input tensor.
 * @param axis The dimension(s) to reduce. If null (the default),
 *     reduces all dimensions.
 * @param keepDims If true, retains reduced dimensions with length
 *     of 1. Defaults to false.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function logSumExp_<T extends Tensor>(
    x: Tensor|TensorLike, axis: number|number[] = null, keepDims = false): T {
  const $x = convertToTensor(x, 'x', 'logSumExp');

  const axes = parseAxisParam(axis, $x.shape);
  const xMax = max($x, axes, true /* keepDims */);
  const a = sub($x, xMax);
  const b = exp(a);
  const c = sum(b, axes);
  const d = log(c);
  const res = add(reshape(xMax, d.shape), d);

  if (keepDims) {
    const newShape = expandShapeToKeepDim(res.shape, axes);
    return reshape(res, newShape) as T;
  }
  return res as T;
}

export const logSumExp = op({logSumExp_});
