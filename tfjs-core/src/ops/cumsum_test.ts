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

describeWithFlags('cumsum', ALL_ENVS, () => {
  it('1D standard', async () => {
    const res = tf.tensor1d([1, 2, 3, 4]).cumsum();
    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [1, 3, 6, 10]);
  });

  it('1D reverse', async () => {
    const reverse = true;
    const exclusive = false;
    const res = tf.tensor1d([1, 2, 3, 4]).cumsum(0, exclusive, reverse);
    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [10, 9, 7, 4]);
  });

  it('1D exclusive', async () => {
    const exclusive = true;
    const res = tf.tensor1d([1, 2, 3, 4]).cumsum(0, exclusive);
    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [0, 1, 3, 6]);
  });

  it('1D exclusive reverse', async () => {
    const reverse = true;
    const exclusive = true;
    const res = tf.tensor1d([1, 2, 3, 4]).cumsum(0, exclusive, reverse);
    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [9, 7, 4, 0]);
  });

  it('gradient: 1D', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([4, 5, 6]);
    const da = tf.grad(x => tf.cumsum(x))(a, dy);

    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [15, 11, 6]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([4, 5, 6]);
    const da = tf.grad(x => tf.cumsum(x.clone()).clone())(a, dy);

    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [15, 11, 6]);
  });

  it('2D standard', async () => {
    const res = tf.tensor2d([[1, 2], [3, 4]]).cumsum(1);
    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(await res.data(), [1, 3, 3, 7]);
  });

  it('2D reverse exclusive', async () => {
    const reverse = true;
    const exclusive = true;
    const res = tf.tensor2d([[1, 2], [3, 4]]).cumsum(1, exclusive, reverse);
    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(await res.data(), [2, 0, 4, 0]);
  });

  it('2D axis=0', async () => {
    const res = tf.tensor2d([[1, 2], [3, 4]]).cumsum();
    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(await res.data(), [1, 2, 4, 6]);
  });

  it('3D standard', async () => {
    const res = tf.tensor3d([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]).cumsum(2);
    expect(res.shape).toEqual([2, 2, 2]);
    expectArraysClose(await res.data(), [0, 1, 2, 5, 4, 9, 6, 13]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.cumsum({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'cumsum' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const res = tf.cumsum([1, 2, 3, 4]);
    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [1, 3, 6, 10]);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.cumsum([
      'a', 'b', 'c'
    ])).toThrowError(/Argument 'x' passed to 'cumsum' must be numeric tensor/);
  });
});
