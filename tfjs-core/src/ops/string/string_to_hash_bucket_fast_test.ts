/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';

describeWithFlags('stringToHashBucketFast', ALL_ENVS, () => {
  it('throw error if negative buckets', async () => {
    expect(() => tf.string.stringToHashBucketFast(['a', 'b', 'c'], -1))
        .toThrowError(/must be at least 1/);
  });

  it('throw error if zero buckets', async () => {
    expect(() => tf.string.stringToHashBucketFast(['a', 'b', 'c'], 0))
        .toThrowError(/must be at least 1/);
  });

  it('one bucket maps values to zero', async () => {
    const result = tf.string.stringToHashBucketFast(['a', 'b', 'c'], 1);
    expectArraysClose(await result.data(), [0, 0, 0]);
  });

  it('multiple buckets', async () => {
    const result = tf.string.stringToHashBucketFast(['a', 'b', 'c', 'd'], 10);
    // fingerPrint64('a') -> 12917804110809363939 -> mod 10 -> 9
    // fingerPrint64('b') -> 11795596070477164822 -> mod 10 -> 2
    // fingerPrint64('c') -> 11430444447143000872 -> mod 10 -> 2
    // fingerPrint64('d') -> 4470636696479570465 -> mod 10 -> 5
    expectArraysClose(await result.data(), [9, 2, 2, 5]);
  });

  it('empty input', async () => {
    const result =
        tf.string.stringToHashBucketFast(tf.tensor1d([], 'string'), 2147483648);
    expectArraysClose(await result.data(), []);
  });

  it('preserve size', async () => {
    const result = tf.string.stringToHashBucketFast(
        [[['a'], ['b']], [['c'], ['d']], [['a'], ['b']]], 10);
    expectArraysClose(await result.data(), [9, 2, 2, 5, 9, 2]);
    expect(result.shape).toEqual([3, 2, 1]);
  });
});
