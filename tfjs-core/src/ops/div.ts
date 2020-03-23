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
import {Div, DivInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {floorDiv} from './binary_ops';
import * as broadcast_util from './broadcast_util';
import {op} from './operation';


/**
 * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting.
 *
 * We also expose `tf.divStrict` which has the same signature as this op and
 * asserts that `a` and `b` are the same shape (does not broadcast).
 *
 * ```js
 * const a = tf.tensor1d([1, 4, 9, 16]);
 * const b = tf.tensor1d([1, 2, 3, 4]);
 *
 * a.div(b).print();  // or tf.div(a, b)
 * ```
 *
 * ```js
 * // Broadcast div a with b.
 * const a = tf.tensor1d([2, 4, 6, 8]);
 * const b = tf.scalar(2);
 *
 * a.div(b).print();  // or tf.div(a, b)
 * ```
 *
 * @param a The first tensor as the numerator.
 * @param b The second tensor as the denominator. Must have the same dtype as
 * `a`.
 */
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function div_<T extends Tensor>(a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'div');
  let $b = convertToTensor(b, 'b', 'div');
  [$a, $b] = makeTypesMatch($a, $b);

  if ($a.dtype === 'int32' && $b.dtype === 'int32') {
    return floorDiv($a, $b);
  }

  const outShape =
      broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
  const der = (dy: Tensor, saved: Tensor[]) => {
    const [$a, $b] = saved;
    const derA = () => {
      const res = dy.div($b.toFloat());
      const reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
      if (reduceAxes.length > 0) {
        return res.sum(reduceAxes).reshape($a.shape);
      }
      return res;
    };
    const derB = () => {
      let res = dy.mul($a.toFloat());
      const reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
      if (reduceAxes.length > 0) {
        res = res.sum(reduceAxes).reshape($b.shape);
      }
      const tmp = $b.square();
      return res.div(tmp.toFloat()).neg();
    };
    return {a: derA, b: derB};
  };

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    const res = backend.realDivide($a, $b);
    save([$a, $b]);
    return res;
  };

  const inputs: DivInputs = {a: $a, b: $b};
  const attrs = {};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, der, Div, attrs) as T;
}

export const div = op({div_});
