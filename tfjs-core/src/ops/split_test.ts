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

describeWithFlags('split', ALL_ENVS, () => {
  it('split by number', async () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const res = tf.split(x, 2, 1);
    expect(res.length).toEqual(2);
    expect(res[0].shape).toEqual([2, 2]);
    expectArraysClose(await res[0].data(), [1, 2, 5, 6]);
    expect(res[1].shape).toEqual([2, 2]);
    expectArraysClose(await res[1].data(), [3, 4, 7, 8]);
  });

  it('split by sizes', async () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const res = tf.split(x, [1, 2, 1], 1);
    expect(res.length).toEqual(3);
    expect(res[0].shape).toEqual([2, 1]);
    expectArraysClose(await res[0].data(), [1, 5]);
    expect(res[1].shape).toEqual([2, 2]);
    expectArraysClose(await res[1].data(), [2, 3, 6, 7]);
    expect(res[2].shape).toEqual([2, 1]);
    expectArraysClose(await res[2].data(), [4, 8]);
  });

  it('chainable split by sizes', async () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const res = x.split([1, 2, 1], 1);

    expect(res.length).toEqual(3);
    expect(res[0].shape).toEqual([2, 1]);
    expectArraysClose(await res[0].data(), [1, 5]);
    expect(res[1].shape).toEqual([2, 2]);
    expectArraysClose(await res[1].data(), [2, 3, 6, 7]);
    expect(res[2].shape).toEqual([2, 1]);
    expectArraysClose(await res[2].data(), [4, 8]);
  });
  it('should support -1 split', async () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const res = x.split([1, 1, -1], 1);

    expect(res.length).toEqual(3);
    expect(res[0].shape).toEqual([2, 1]);
    expectArraysClose(await res[0].data(), [1, 5]);
    expect(res[1].shape).toEqual([2, 1]);
    expectArraysClose(await res[1].data(), [2, 6]);
    expect(res[2].shape).toEqual([2, 2]);
    expectArraysClose(await res[2].data(), [3, 4, 7, 8]);
  });

  it('multiple negative number throws error', () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const f = () => tf.split(x, [1, -1, -1], 1);
    expect(f).toThrowError();
  });
  it('sizes to not sum to axis size throws error', () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const f = () => tf.split(x, [1, 2], 1);
    expect(f).toThrowError();
  });

  it('number of splits does not evenly divide axis', () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const f = () => tf.split(x, 3, 1);
    expect(f).toThrowError();
  });

  it('can split, axis=-2', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const numSplits = 2;
    const axis = -2;
    const res = tf.split(a, numSplits, axis);
    expect(res.length).toBe(2);
    expect(res[0].shape).toEqual([2, 1, 2]);
    expect(res[1].shape).toEqual([2, 1, 2]);
    expectArraysClose(await res[0].data(), [1, 2, 5, 6]);
    expectArraysClose(await res[1].data(), [3, 4, 7, 8]);
  });

  it('can split a zero-sized tensor, axis=0', async () => {
    const a = tf.zeros([4, 0]);
    const numSplits = 4;
    const axis = 0;
    const res = tf.split(a, numSplits, axis);
    expect(res.length).toBe(4);
    expect(res[0].shape).toEqual([1, 0]);
    expect(res[1].shape).toEqual([1, 0]);
    expect(res[2].shape).toEqual([1, 0]);
    expect(res[3].shape).toEqual([1, 0]);
    expectArraysClose(await res[0].data(), []);
    expectArraysClose(await res[1].data(), []);
    expectArraysClose(await res[2].data(), []);
    expectArraysClose(await res[3].data(), []);
  });

  it('can split a zero-sized tensor, axis=1', async () => {
    const a = tf.zeros([0, 4]);
    const numSplits = 4;
    const axis = 1;
    const res = tf.split(a, numSplits, axis);
    expect(res.length).toBe(4);
    expect(res[0].shape).toEqual([0, 1]);
    expect(res[1].shape).toEqual([0, 1]);
    expect(res[2].shape).toEqual([0, 1]);
    expect(res[3].shape).toEqual([0, 1]);
    expectArraysClose(await res[0].data(), []);
    expectArraysClose(await res[1].data(), []);
    expectArraysClose(await res[2].data(), []);
    expectArraysClose(await res[3].data(), []);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.split({} as tf.Tensor, 1))
        .toThrowError(/Argument 'x' passed to 'split' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const x = [[1, 2, 3, 4], [5, 6, 7, 8]];
    const res = tf.split(x, 2, 1);
    expect(res.length).toEqual(2);
    expect(res[0].shape).toEqual([2, 2]);
    expectArraysClose(await res[0].data(), [1, 2, 5, 6]);
    expect(res[1].shape).toEqual([2, 2]);
    expectArraysClose(await res[1].data(), [3, 4, 7, 8]);
  });

  it('gradient of 1st output', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const da = tf.grad(x => tf.split(x, [1, 2])[0])(a);

    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [1, 0, 0]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const da = tf.grad(x => tf.split(x.clone(), [1, 2])[0].clone())(a);

    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [1, 0, 0]);
  });

  it('gradient of 2nd output', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const da = tf.grad(x => tf.split(x, [1, 2])[1])(a);

    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [0, 1, 1]);
  });
});
