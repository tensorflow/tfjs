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
import {ENGINE, ForwardFunc} from '../engine';
import {SplitV, SplitVAttrs, SplitVInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {assert,} from '../util';
import {parseAxisParam} from '../util';

import {op} from './operation';

/**
 * Splits a `tf.Tensor` into sub tensors.
 *
 * If `numOrSizeSplits` is a number, splits `x` along dimension `axis`
 * into `numOrSizeSplits` smaller tensors.
 * Requires that `numOrSizeSplits` evenly divides `x.shape[axis]`.
 *
 * If `numOrSizeSplits` is a number array, splits `x` into
 * `numOrSizeSplits.length` pieces. The shape of the `i`-th piece has the
 * same size as `x` except along dimension `axis` where the size is
 * `numOrSizeSplits[i]`.
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
 * const [a, b] = tf.split(x, 2, 1);
 * a.print();
 * b.print();
 *
 * const [c, d, e] = tf.split(x, [1, 2, 1], 1);
 * c.print();
 * d.print();
 * e.print();
 * ```
 *
 * @param x The input tensor to split.
 * @param numOrSizeSplits Either an integer indicating the number of
 * splits along the axis or an array of integers containing the sizes of
 * each output tensor along the axis. If a number then it must evenly divide
 * `x.shape[axis]`; otherwise the sum of sizes must match `x.shape[axis]`.
 * @param axis The dimension along which to split. Defaults to 0 (the first
 * dim).
 */
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function split_<T extends Tensor>(
    x: Tensor|TensorLike, numOrSizeSplits: number[]|number, axis = 0): T[] {
  const $x = convertToTensor(x, 'x', 'split');

  const $axis = parseAxisParam(axis, $x.shape)[0];
  let splitSizes: number[];

  if (typeof (numOrSizeSplits) === 'number') {
    assert(
        $x.shape[$axis] % numOrSizeSplits === 0,
        () => 'Number of splits must evenly divide the axis.');
    splitSizes =
        new Array(numOrSizeSplits).fill($x.shape[$axis] / numOrSizeSplits);
  } else {
    assert(
        $x.shape[$axis] === numOrSizeSplits.reduce((a, b) => a + b),
        () => 'The sum of sizes must match the size of the axis dimension.');
    splitSizes = numOrSizeSplits;
  }

  const forward: ForwardFunc<Tensor> = (backend, _) => {
    return backend.split($x, splitSizes, $axis) as {} as T;
  };

  const inputs: SplitVInputs = {x: $x};
  const attr: SplitVAttrs = {numOrSizeSplits, axis};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */, SplitV,
             attr as {} as NamedAttrMap) as {} as T[];
}

export const split = op({split_});
