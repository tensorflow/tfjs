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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

describeWithFlags('reverse1d', ALL_ENVS, () => {
  it('reverse a 1D array', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const result = tf.reverse1d(input);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [5, 4, 3, 2, 1]);
  });

  it('reverse a 1D array, even length', async () => {
    const input = tf.tensor1d([1, 2, 3, 4]);
    const result = tf.reverse1d(input);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [4, 3, 2, 1]);
  });

  it('grad', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([10, 20, 30]);
    const da = tf.grad((a: tf.Tensor1D) => tf.reverse1d(a))(a, dy);
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [30, 20, 10]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([10, 20, 30]);
    const da =
        tf.grad((a: tf.Tensor1D) => tf.reverse1d(a.clone()).clone())(a, dy);
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [30, 20, 10]);
  });

  it('accepts a tensor-like object', async () => {
    const input = [1, 2, 3, 4, 5];
    const result = tf.reverse1d(input);
    expect(result.shape).toEqual([5]);
    expectArraysClose(await result.data(), [5, 4, 3, 2, 1]);
  });
});
