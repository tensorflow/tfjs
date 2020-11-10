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
import {KernelBackend} from '../backends/backend';
import {ENGINE, ForwardFunc} from '../engine';
import {Min, MinAttrs, MinInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {GradSaveFunc, NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {parseAxisParam} from '../util';

import * as axis_util from './axis_util';
import {op} from './operation';
import {reshape} from './reshape';
import {transpose} from './transpose';

/**
 * Computes the minimum value from the input.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the array is reduced by 1 for each entry in `axes`.
 * If `keepDims` is true, the reduced dimensions are retained with length 1.
 * If `axes` has no entries, all dimensions are reduced, and an array with a
 * single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.min().print();  // or tf.min(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.min(axis).print();  // or tf.min(x, axis)
 * ```
 *
 * @param x The input Tensor.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function min_<T extends Tensor>(
    x: Tensor|TensorLike, axis: number|number[] = null, keepDims = false): T {
  const $x = convertToTensor(x, 'x', 'min');

  const forward: ForwardFunc<Tensor> =
      (backend: KernelBackend, save: GradSaveFunc) => {
        const origAxes = parseAxisParam(axis, $x.shape);
        let axes = origAxes;
        const permutedAxes = axis_util.getAxesPermutation(axes, $x.rank);
        let minInput = $x;
        if (permutedAxes != null) {
          minInput = transpose($x, permutedAxes);
          axes = axis_util.getInnerMostAxes(axes.length, $x.rank);
        }

        const y = backend.min(minInput, axes);
        if (permutedAxes != null) {
          minInput.dispose();
        }

        let res = y;
        if (keepDims) {
          const expandedShape =
              axis_util.expandShapeToKeepDim(res.shape, origAxes);
          res = reshape(y, expandedShape) as T;
          y.dispose();
        }

        save([$x, res]);
        return res;
      };

  const inputs: MinInputs = {x: $x};
  const attrs: MinAttrs = {axis, keepDims};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* gradient */, Min,
             attrs as {} as NamedAttrMap) as T;
}

export const min = op({min_});
