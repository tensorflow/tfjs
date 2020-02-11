/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import {customGrad} from '../gradients';
import {Tensor} from '../tensor';
import {GradSaveFunc} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {op} from './operation';

/**
 * Computes the softmax normalized vector given the logits.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 *
 * a.softmax().print();  // or tf.softmax(a)
 * ```
 *
 * ```js
 * const a = tf.tensor2d([2, 4, 6, 1, 2, 3], [2, 3]);
 *
 * a.softmax().print();  // or tf.softmax(a)
 * ```
 *
 * @param logits The logits array.
 * @param dim The dimension softmax would be performed on. Defaults to `-1`
 *     which indicates the last dimension.
 */
/** @doc {heading: 'Operations', subheading: 'Normalization'} */
function softmax_<T extends Tensor>(logits: T|TensorLike, dim = -1): T {
  const $logits = convertToTensor(logits, 'logits', 'softmax', 'float32');

  if (dim === -1) {
    dim = $logits.rank - 1;
  }
  if (dim !== $logits.rank - 1) {
    throw Error(
        'Softmax along a non-last dimension is not yet supported. ' +
        `Logits was rank ${$logits.rank} and dim was ${dim}`);
  }

  const inputsToSave: Tensor[] = [];
  const outputsToSave = [true];

  return ENGINE.runKernelFunc(
      (backend, save) => {
        const y = backend.softmax($logits, dim);
        save([y]);
        return y;
      },
      {logits: $logits},
      (dy: T, saved: Tensor[]) => {
        const [y] = saved;
        const dyTimesY = dy.mul(y);
        const keepDims = true;

        return {
          logits: () => dyTimesY.sub(dyTimesY.sum([dim], keepDims).mul(y))
        };
      },
      'Softmax', {dim}, inputsToSave, outputsToSave);
}

/**
 * Computes the log softmax.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 *
 * a.logSoftmax().print();  // or tf.logSoftmax(a)
 * ```
 *
 * ```js
 * const a = tf.tensor2d([2, 4, 6, 1, 2, 3], [2, 3]);
 *
 * a.logSoftmax().print();  // or tf.logSoftmax(a)
 * ```
 *
 * @param logits The logits array.
 * @param axis The dimension softmax would be performed on. Defaults to `-1`
 *     which indicates the last dimension.
 */
/** @doc {heading: 'Operations', subheading: 'Normalization'} */
function logSoftmax_<T extends Tensor>(logits: T|TensorLike, axis = -1): T {
  const $logits = convertToTensor(logits, 'logits', 'logSoftmax');

  if (axis === -1) {
    axis = $logits.rank - 1;
  }
  if (axis !== $logits.rank - 1) {
    throw Error(
        'Log Softmax along a non-last dimension is not yet supported. ' +
        `Logits was rank ${$logits.rank} and axis was ${axis}`);
  }

  const customOp = customGrad((logits: Tensor, save: GradSaveFunc) => {
    const keepDims = true;
    const xMax = logits.max(axis, true);
    const shifted = logits.sub(xMax);
    const value =
        shifted.toFloat().sub(shifted.exp().sum(axis, keepDims).log());
    save([value]);
    const gradFunc = (dy: T, saved: Tensor[]) => {
      const [value] = saved;
      const softmax = value.exp();
      return dy.sub(dy.sum(axis, keepDims).mul(softmax));
    };

    return {value, gradFunc};
  });

  return customOp($logits) as T;
}

export const softmax = op({softmax_});
export const logSoftmax = op({logSoftmax_});
