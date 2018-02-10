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

import {ENV} from '../environment';
import * as util from '../util';

import * as broadcast_util from './broadcast_util';
import {doc, operation} from './decorators';
import {Tensor} from './tensor';

export class Ops {
  /**
   * Returns the truth value of (a != b) element-wise. Supports broadcasting.
   * For a stricter version without broadcasting use dl.notEqualStrict().
   *
   * @param a The first input `Tensor`.
   * @param b The second input `Tensor`. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static notEqual<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('NotEqual', {inputs: {a, b}}) as T;
  }

  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static notEqualStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in notEqualStrict: ');
    return a.notEqual(b);
  }

  /**
   * Returns the truth value of (a < b) element-wise. Supports broadcasting.
   * For a stricter version without broadcasting use dl.lessStrict().
   *
   * @param a The first input `Tensor`.
   * @param b The second input `Tensor`. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static less<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('Less', {inputs: {a, b}}) as T;
  }

  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static lessStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in lessStrict: ');
    return a.less(b);
  }

  /**
   * Returns the truth value of (a == b) element-wise. Supports broadcasting.
   * For a stricter version without broadcasting use dl.equalStrict().
   *
   * @param a The first input `Tensor`.
   * @param b The second input `Tensor`. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static equal<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('Equal', {inputs: {a, b}}) as T;
  }

  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static equalStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in equalStrict: ');
    return a.equal(b);
  }

  /**
   * Returns the truth value of (a <= b) element-wise. Supports broadcasting.
   * For a stricter version without broadcasting use dl.lessEqualStrict().
   *
   * @param a The first input `Tensor`.
   * @param b The second input `Tensor`. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static lessEqual<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('LessEqual', {inputs: {a, b}}) as T;
  }

  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static lessEqualStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in lessEqualStrict: ');
    return a.lessEqual(b);
  }

  /**
   * Returns the truth value of (a > b) element-wise. Supports broadcasting.
   * For a stricter version without broadcasting use dl.greaterStrict().
   *
   * @param a The first input `Tensor`.
   * @param b The second input `Tensor`. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static greater<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('Greater', {inputs: {a, b}}) as T;
  }

  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static greaterStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in greaterStrict: ');
    return a.greater(b);
  }

  /**
   * Returns the truth value of (a >= b) element-wise. Supports broadcasting.
   * For a stricter version without broadcasting use dl.greaterEqualStrict().
   *
   * @param a The first input `Tensor`.
   * @param b The second input `Tensor`. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static greaterEqual<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('GreaterEqual', {inputs: {a, b}}) as T;
  }

  @doc({heading: 'Operations', subheading: 'Logical'})
  @operation
  static greaterEqualStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in greaterEqualStrict: ');
    return a.greaterEqual(b);
  }
}
