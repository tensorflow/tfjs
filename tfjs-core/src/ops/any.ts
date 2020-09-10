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

import {ENGINE, ForwardFunc} from '../engine';
import {Any, AnyAttrs, AnyInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {parseAxisParam} from '../util';

import {expandShapeToKeepDim, getAxesPermutation, getInnerMostAxes} from './axis_util';
import {op} from './operation';
import {reshape} from './reshape';
import {transpose} from './transpose';

/**
 * Computes the logical or of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
 * `axes`. If `keepDims` is true, the reduced dimensions are retained with
 * length 1. If `axes` has no entries, all dimensions are reduced, and an
 * `tf.Tensor` with a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 1, 1], 'bool');
 *
 * x.any().print();  // or tf.any(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
 *
 * const axis = 1;
 * x.any(axis).print();  // or tf.any(x, axis)
 * ```
 *
 * @param x The input tensor. Must be of dtype bool.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function any_<T extends Tensor>(
    x: Tensor|TensorLike, axis: number|number[] = null, keepDims = false): T {
  let $x = convertToTensor(x, 'x', 'any', 'bool');

  const forward: ForwardFunc<Tensor> = (backend) => {
    const origAxes = parseAxisParam(axis, $x.shape);
    let axes = origAxes;
    const permutedAxes = getAxesPermutation(axes, $x.rank);
    if (permutedAxes != null) {
      $x = transpose($x, permutedAxes);
      axes = getInnerMostAxes(axes.length, $x.rank);
    }
    const res = backend.any($x, axes);
    if (keepDims) {
      const newShape = expandShapeToKeepDim(res.shape, origAxes);
      return reshape(res, newShape);
    }
    return res as T;
  };

  const inputs: AnyInputs = {x: $x};
  const attrs: AnyAttrs = {axis, keepDims};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */, Any,
             attrs as {} as NamedAttrMap) as T;
}

// tslint:disable-next-line:variable-name
export const any = op({any_});
