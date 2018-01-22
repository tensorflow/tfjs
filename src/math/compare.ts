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
import {operation} from './decorators';
import {DataType, NDArray, Rank, RankMap} from './ndarray';

export class Ops {
  /**
   * Returns the truth value of (a != b) element-wise. Supports broadcasting.
   * For a stricter version without broadcasting use math.notEqualStrict().
   *
   * @param a The first input `NDArray`.
   * @param b The second input `NDArray`. Must have the same dtype as `a`.
   */
  @operation
  static notEqual<D1 extends DataType, D2 extends D1, T extends
                      NDArray<'bool'>>(a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('NotEqual', {inputs: {a, b}}) as T;
  }

  @operation
  static notEqualStrict<R extends Rank, D1 extends DataType, D2 extends D1>(
      a: NDArray<D1, R>, b: NDArray<D2, R>): RankMap<'bool'>[R] {
    util.assertShapesMatch(a.shape, b.shape, 'Error in notEqualStrict: ');
    return Ops.notEqual(a, b);
  }

  /**
   * Returns the truth value of (a < b) element-wise. Supports broadcasting.
   *
   * @param a The first input `NDArray`.
   * @param b The second input `NDArray`. Must have the same dtype as `a`.
   */
  @operation
  static less<D1 extends DataType, D2 extends D1, T extends NDArray<'bool'>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('Less', {inputs: {a, b}}) as T;
  }

  /**
   * Returns the truth value of (a == b) element-wise. Supports broadcasting.
   * For a stricter version without broadcasting use math.equalStrict().
   *
   * @param a The first input `NDArray`.
   * @param b The second input `NDArray`. Must have the same dtype as `a`.
   */
  @operation
  static equal<D1 extends DataType, D2 extends D1, T extends NDArray<'bool'>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('Equal', {inputs: {a, b}}) as T;
  }

  @operation
  static equalStrict<T extends NDArray>(a: T, b: T): NDArray<'bool'> {
    util.assertShapesMatch(a.shape, b.shape, 'Error in equalStrict: ');
    return Ops.equal(a, b);
  }

  /**
   * Returns the truth value of (a <= b) element-wise. Supports broadcasting.
   *
   * @param a The first input `NDArray`.
   * @param b The second input `NDArray`. Must have the same dtype as `a`.
   */
  @operation
  static lessEqual<D1 extends DataType, D2 extends D1, T extends
                       NDArray<'bool'>>(a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('LessEqual', {inputs: {a, b}}) as T;
  }

  /**
   * Returns the truth value of (a > b) element-wise. Supports broadcasting.
   *
   * @param a The first input `NDArray`.
   * @param b The second input `NDArray`. Must have the same dtype as `a`.
   */
  @operation
  static greater<D1 extends DataType, D2 extends D1, T extends NDArray<'bool'>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('Greater', {inputs: {a, b}}) as T;
  }

  /**
   * Returns the truth value of (a >= b) element-wise. Supports broadcasting.
   *
   * @param a The first input `NDArray`.
   * @param b The second input `NDArray`. Must have the same dtype as `a`.
   */
  @operation
  static greaterEqual<D1 extends DataType, D2 extends D1, T extends
                          NDArray<'bool'>>(a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('GreaterEqual', {inputs: {a, b}}) as T;
  }
}
