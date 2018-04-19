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

import {doc} from '../doc';
import {ENV} from '../environment';
import {Tensor} from '../tensor';
import * as util from '../util';
import * as broadcast_util from './broadcast_util';
import {operation} from './operation';

export class CompareOps {
  /**
   * Returns the truth value of (a != b) element-wise. Supports broadcasting.
   *
   * We also expose `notEqualStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([0, 2, 3]);
   *
   * a.notEqual(b).print();
   * ```
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static notEqual<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'notEqual');
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    return ENV.engine.runKernel(backend => backend.notEqual(a, b), {a, b}) as T;
  }

  /**
   * Strict version of `notEqual` that forces `a` and `b` to be of the same
   * shape.
   *
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same shape and dtype as
   *     `a`.
   */
  @operation
  static notEqualStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in notEqualStrict: ');
    return a.notEqual(b);
  }

  /**
   * Returns the truth value of (a < b) element-wise. Supports broadcasting.
   *
   * We also expose `lessStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([2, 2, 2]);
   *
   * a.less(b).print();
   * ```
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static less<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'less');
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    return ENV.engine.runKernel(backend => backend.less(a, b), {a, b}) as T;
  }

  /**
   * Strict version of `less` that forces `a` and `b` to be of the same
   * shape.
   *
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same shape and dtype as
   *     `a`.
   */
  @operation
  static lessStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in lessStrict: ');
    return a.less(b);
  }

  /**
   * Returns the truth value of (a == b) element-wise. Supports broadcasting.
   *
   * We also expose `equalStrict` which has the same signature as this op
   * and asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([2, 2, 2]);
   *
   * a.equal(b).print();
   * ```
   *
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static equal<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'equal');
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    return ENV.engine.runKernel(backend => backend.equal(a, b), {a, b}) as T;
  }

  @operation
  static equalStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in equalStrict: ');
    return a.equal(b);
  }

  /**
   * Returns the truth value of (a <= b) element-wise. Supports broadcasting.
   *
   * We also expose `lessEqualStrict` which has the same signature as this op
   * and asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([2, 2, 2]);
   *
   * a.lessEqual(b).print();
   * ```
   *
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static lessEqual<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'lessEqual');
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    return ENV.engine.runKernel(backend => backend.lessEqual(a, b), {a, b}) as
        T;
  }

  @operation
  static lessEqualStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in lessEqualStrict: ');
    return a.lessEqual(b);
  }

  /**
   * Returns the truth value of (a > b) element-wise. Supports broadcasting.
   *
   * We also expose `greaterStrict` which has the same signature as this
   * op and asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([2, 2, 2]);
   *
   * a.greater(b).print();
   * ```
   *
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static greater<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'greater');
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    return ENV.engine.runKernel(backend => backend.greater(a, b), {a, b}) as T;
  }

  @operation
  static greaterStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in greaterStrict: ');
    return a.greater(b);
  }

  /**
   * Returns the truth value of (a >= b) element-wise. Supports broadcasting.
   *
   * We also expose `greaterEqualStrict` which has the same signature as this
   * op and asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([2, 2, 2]);
   *
   * a.greaterEqual(b).print();
   * ```
   *
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static greaterEqual<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'greaterEqual');
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    return ENV.engine.runKernel(
               backend => backend.greaterEqual(a, b), {a, b}) as T;
  }

  @operation
  static greaterEqualStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in greaterEqualStrict: ');
    return a.greaterEqual(b);
  }
}
