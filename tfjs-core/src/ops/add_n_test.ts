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

describeWithFlags('addN', ALL_ENVS, () => {
  it('a single tensor', async () => {
    const res = tf.addN([tf.tensor1d([1, 2, 3])]);
    expectArraysClose(await res.data(), [1, 2, 3]);
  });

  it('two tensors, int32', async () => {
    const res = tf.addN([
      tf.tensor1d([1, 2, -1], 'int32'),
      tf.tensor1d([5, 3, 2], 'int32'),
    ]);
    expectArraysClose(await res.data(), [6, 5, 1]);
    expect(res.dtype).toBe('int32');
    expect(res.shape).toEqual([3]);
  });

  it('three tensors', async () => {
    const res = tf.addN([
      tf.tensor1d([1, 2]),
      tf.tensor1d([5, 3]),
      tf.tensor1d([-5, -2]),
    ]);
    expectArraysClose(await res.data(), [1, 3]);
    expect(res.dtype).toBe('float32');
    expect(res.shape).toEqual([2]);
  });

  it('accepts a tensor-like object', async () => {
    const res = tf.addN([[1, 2], [3, 4]]);
    expectArraysClose(await res.data(), [4, 6]);
    expect(res.dtype).toBe('float32');
    expect(res.shape).toEqual([2]);
  });

  it('list of numbers gets treated as a list of scalars', async () => {
    const res = tf.addN([1, 2, 3, 4]);
    expectArraysClose(await res.data(), [10]);
    expect(res.dtype).toBe('float32');
    expect(res.shape).toEqual([]);
  });

  it('errors if list is empty', () => {
    expect(() => tf.addN([]))
        .toThrowError(
            /Must pass at least one tensor to tf.addN\(\), but got 0/);
  });

  it('errors if argument is not an array', () => {
    // tslint:disable-next-line:no-any
    expect(() => tf.addN(tf.scalar(3) as any))
        .toThrowError(
            /The argument passed to tf.addN\(\) must be a list of tensors/);
  });

  it('errors if arguments not of same dtype', () => {
    expect(() => tf.addN([tf.scalar(1, 'int32'), tf.scalar(2, 'float32')]))
        .toThrowError(
            /All tensors passed to tf.addN\(\) must have the same dtype/);
  });

  it('errors if arguments not of same shape', () => {
    expect(() => tf.addN([tf.scalar(1), tf.tensor1d([2])]))
        .toThrowError(
            /All tensors passed to tf.addN\(\) must have the same shape/);
  });
});
