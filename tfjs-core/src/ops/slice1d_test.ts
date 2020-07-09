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

describeWithFlags('slice1d', ALL_ENVS, () => {
  it('slices 1x1 into 1x1 (effectively a copy)', async () => {
    const a = tf.tensor1d([5]);
    const result = tf.slice1d(a, 0, 1);

    expect(result.shape).toEqual([1]);
    expectArraysClose(await result.data(), 5);
  });

  it('slices 5x1 into shape 2x1 starting at 3', async () => {
    const a = tf.tensor1d([1, 2, 3, 4, 5]);
    const result = tf.slice1d(a, 3, 2);

    expect(result.shape).toEqual([2]);
    expectArraysClose(await result.data(), [4, 5]);
  });

  it('slices 5x1 into shape 3x1 starting at 1', async () => {
    const a = tf.tensor1d([1, 2, 3, 4, 5]);
    const result = tf.slice1d(a, 1, 3);

    expect(result.shape).toEqual([3]);
    expectArraysClose(await result.data(), [2, 3, 4]);
  });

  it('grad', async () => {
    const a = tf.tensor1d([1, 2, 3, 4, 5]);
    const dy = tf.tensor1d([10, 100]);
    const da = tf.grad((a: tf.Tensor1D) => tf.slice1d(a, 1, 2))(a, dy);
    expect(da.shape).toEqual([5]);
    expectArraysClose(await da.data(), [0, 10, 100, 0, 0]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([1, 2, 3, 4, 5]);
    const dy = tf.tensor1d([10, 100]);
    const da =
        tf.grad((a: tf.Tensor1D) => tf.slice1d(a.clone(), 1, 2).clone())(a, dy);
    expect(da.shape).toEqual([5]);
    expectArraysClose(await da.data(), [0, 10, 100, 0, 0]);
  });

  it('accepts a tensor-like object', async () => {
    const a = [5];
    const result = tf.slice1d(a, 0, 1);
    expect(result.shape).toEqual([1]);
    expectArraysClose(await result.data(), 5);
  });
});
