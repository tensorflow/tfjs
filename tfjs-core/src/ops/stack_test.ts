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

describeWithFlags('stack', ALL_ENVS, () => {
  it('scalars 3, 5 and 7', async () => {
    const a = tf.scalar(3);
    const b = tf.scalar(5);
    const c = tf.scalar(7);
    const res = tf.stack([a, b, c]);
    expect(res.shape).toEqual([3]);
    expectArraysClose(await res.data(), [3, 5, 7]);
  });

  it('scalars 3, 5 and 7 along axis=1 throws error', () => {
    const a = tf.scalar(3);
    const b = tf.scalar(5);
    const c = tf.scalar(7);
    const f = () => tf.stack([a, b, c], 1);
    expect(f).toThrowError();
  });

  it('non matching shapes throws error', () => {
    const a = tf.scalar(3);
    const b = tf.tensor1d([5]);
    const f = () => tf.stack([a, b]);
    expect(f).toThrowError();
  });

  it('non matching dtypes throws error', () => {
    const a = tf.scalar(3);
    const b = tf.scalar(5, 'bool');
    const f = () => tf.stack([a, b]);
    expect(f).toThrowError();
  });

  it('2d but axis=3 throws error', () => {
    const a = tf.zeros([2, 2]);
    const b = tf.zeros([2, 2]);
    const f = () => tf.stack([a, b], 3 /* axis */);
    expect(f).toThrowError();
  });

  it('[1,2], [3,4] and [5,6], axis=0', async () => {
    const a = tf.tensor1d([1, 2]);
    const b = tf.tensor1d([3, 4]);
    const c = tf.tensor1d([5, 6]);
    const res = tf.stack([a, b, c], 0 /* axis */);
    expect(res.shape).toEqual([3, 2]);
    expectArraysClose(await res.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('[1,2], [3,4] and [5,6], axis=1', async () => {
    const a = tf.tensor1d([1, 2]);
    const b = tf.tensor1d([3, 4]);
    const c = tf.tensor1d([5, 6]);
    const res = tf.stack([a, b, c], 1 /* axis */);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(await res.data(), [1, 3, 5, 2, 4, 6]);
  });

  it('[[1,2],[3,4]] and [[5, 6], [7, 8]], axis=0', async () => {
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const b = tf.tensor2d([[5, 6], [7, 8]]);
    const res = tf.stack([a, b], 0 /* axis */);
    expect(res.shape).toEqual([2, 2, 2]);
    expectArraysClose(await res.data(), [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('[[1,2],[3,4]] and [[5, 6], [7, 8]], axis=2', async () => {
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const b = tf.tensor2d([[5, 6], [7, 8]]);
    const c = tf.tensor2d([[9, 10], [11, 12]]);
    const res = tf.stack([a, b, c], 2 /* axis */);
    expect(res.shape).toEqual([2, 2, 3]);
    expectArraysClose(
        await res.data(), [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]);
  });

  it('single tensor', async () => {
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const res = tf.stack([a], 2 /* axis */);
    expect(res.shape).toEqual([2, 2, 1]);
    expectArraysClose(await res.data(), [1, 2, 3, 4]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.stack([{} as tf.Tensor]))
        .toThrowError(
            /Argument 'tensors\[0\]' passed to 'stack' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [[1, 2], [3, 4]];
    const res = tf.stack([a], 2 /* axis */);
    expect(res.shape).toEqual([2, 2, 1]);
    expectArraysClose(await res.data(), [1, 2, 3, 4]);
  });

  it('accepts string.', async () => {
    const a = tf.scalar('three', 'string');
    const b = tf.scalar('five', 'string');
    const c = tf.scalar('seven', 'string');
    const res = tf.stack([a, b, c]);
    expect(res.shape).toEqual([3]);
    expectArraysClose(await res.data(), ['three', 'five', 'seven']);
  });

  it('chain api', async () => {
    const a = tf.tensor([1, 2]);
    const res = a.stack(tf.tensor([3, 4]));
    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(await res.data(), [1, 2, 3, 4]);
  });
});
