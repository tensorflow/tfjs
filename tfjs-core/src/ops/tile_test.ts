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
import {expectArraysClose, expectArraysEqual} from '../test_util';

describeWithFlags('tile', ALL_ENVS, () => {
  it('1D (tile)', async () => {
    const t = tf.tensor1d([1, 2, 3]);
    const t2 = tf.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expectArraysClose(await t2.data(), [1, 2, 3, 1, 2, 3]);
  });

  it('2D (tile)', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    let t2 = tf.tile(t, [1, 2]);

    expect(t2.shape).toEqual([2, 4]);
    expectArraysClose(await t2.data(), [1, 11, 1, 11, 2, 22, 2, 22]);

    t2 = tf.tile(t, [2, 1]);
    expect(t2.shape).toEqual([4, 2]);
    expectArraysClose(await t2.data(), [1, 11, 2, 22, 1, 11, 2, 22]);

    t2 = tf.tile(t, [2, 2]);
    expect(t2.shape).toEqual([4, 4]);
    expectArraysClose(
        await t2.data(),
        [1, 11, 1, 11, 2, 22, 2, 22, 1, 11, 1, 11, 2, 22, 2, 22]);
  });

  it('3D (tile)', async () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const t2 = tf.tile(t, [1, 2, 1]);

    expect(t2.shape).toEqual([2, 4, 2]);
    expectArraysClose(
        await t2.data(), [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
  });

  it('4D (tile)', async () => {
    const t = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2]);
    const t2 = tf.tile(t, [1, 2, 1, 1]);

    expect(t2.shape).toEqual([1, 4, 2, 2]);
    expectArraysClose(
        await t2.data(), [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('5D (tile)', async () => {
    const t = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 2, 2]);
    const t2 = tf.tile(t, [1, 2, 1, 1, 1]);

    expect(t2.shape).toEqual([1, 2, 2, 2, 2]);
    expectArraysClose(
        await t2.data(), [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('1d string tensor', async () => {
    const a = tf.tensor(['a', 'b', 'c']);
    const res = tf.tile(a, [2]);
    expect(res.shape).toEqual([6]);
    expectArraysEqual(await res.data(), ['a', 'b', 'c', 'a', 'b', 'c']);
  });

  it('2d string tensor', async () => {
    const a = tf.tensor([['a', 'b'], ['c', 'd']]);
    const res = tf.tile(a, [2, 3]);
    expect(res.shape).toEqual([4, 6]);
    expectArraysEqual(await res.data(), [
      'a', 'b', 'a', 'b', 'a', 'b', 'c', 'd', 'c', 'd', 'c', 'd',
      'a', 'b', 'a', 'b', 'a', 'b', 'c', 'd', 'c', 'd', 'c', 'd'
    ]);
  });

  it('propagates NaNs', async () => {
    const t = tf.tensor1d([1, 2, NaN]);

    const t2 = tf.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expectArraysClose(await t2.data(), [1, 2, NaN, 1, 2, NaN]);
  });

  it('1D bool (tile)', async () => {
    const t = tf.tensor1d([true, false, true], 'bool');
    const t2 = tf.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(await t2.data(), [1, 0, 1, 1, 0, 1]);
  });

  it('2D bool (tile)', async () => {
    const t = tf.tensor2d([true, false, true, true], [2, 2], 'bool');
    let t2 = tf.tile(t, [1, 2]);

    expect(t2.shape).toEqual([2, 4]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(await t2.data(), [1, 0, 1, 0, 1, 1, 1, 1]);

    t2 = tf.tile(t, [2, 1]);
    expect(t2.shape).toEqual([4, 2]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(await t2.data(), [1, 0, 1, 1, 1, 0, 1, 1]);

    t2 = tf.tile(t, [2, 2]);
    expect(t2.shape).toEqual([4, 4]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(
        await t2.data(), [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]);
  });

  it('3D bool (tile)', async () => {
    const t = tf.tensor3d(
        [true, false, true, false, true, false, true, false], [2, 2, 2],
        'bool');
    const t2 = tf.tile(t, [1, 2, 1]);

    expect(t2.shape).toEqual([2, 4, 2]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(
        await t2.data(), [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
  });

  it('1D int32 (tile)', async () => {
    const t = tf.tensor1d([1, 2, 5], 'int32');
    const t2 = tf.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(await t2.data(), [1, 2, 5, 1, 2, 5]);
  });

  it('2D int32 (tile)', async () => {
    const t = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    let t2 = tf.tile(t, [1, 2]);

    expect(t2.shape).toEqual([2, 4]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(await t2.data(), [1, 2, 1, 2, 3, 4, 3, 4]);

    t2 = tf.tile(t, [2, 1]);
    expect(t2.shape).toEqual([4, 2]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(await t2.data(), [1, 2, 3, 4, 1, 2, 3, 4]);

    t2 = tf.tile(t, [2, 2]);
    expect(t2.shape).toEqual([4, 4]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(
        await t2.data(), [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4]);
  });

  it('3D int32 (tile)', async () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2], 'int32');
    const t2 = tf.tile(t, [1, 2, 1]);

    expect(t2.shape).toEqual([2, 4, 2]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(
        await t2.data(), [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
  });

  it('1D (tile) gradient', async () => {
    const x = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([0.1, 0.2, 0.3, 1, 2, 3, 10, 20, 30]);
    const gradients = tf.grad(x => tf.tile(x, [3]))(x, dy);
    expectArraysClose(await gradients.data(), [11.1, 22.2, 33.3]);
    expect(gradients.shape).toEqual([3]);
  });

  it('gradient with clones', async () => {
    const x = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([0.1, 0.2, 0.3, 1, 2, 3, 10, 20, 30]);
    const gradients = tf.grad(x => tf.tile(x.clone(), [3]).clone())(x, dy);
    expectArraysClose(await gradients.data(), [11.1, 22.2, 33.3]);
    expect(gradients.shape).toEqual([3]);
  });

  it('2D (tile) gradient', async () => {
    const x = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const dy = tf.tensor2d([[1, 2, 10, 20], [3, 4, 30, 40]], [2, 4]);
    const gradients = tf.grad(x => tf.tile(x, [1, 2]))(x, dy);
    expectArraysClose(await gradients.data(), [11, 22, 33, 44]);
    expect(gradients.shape).toEqual([2, 2]);
  });

  it('3D (tile) gradient', async () => {
    const x = tf.tensor3d([[[1], [2]], [[3], [4]]], [2, 2, 1]);
    const dy = tf.tensor3d([[[1, 10], [2, 20]], [[3, 30], [4, 40]]], [2, 2, 2]);
    const gradients = tf.grad(x => tf.tile(x, [1, 1, 2]))(x, dy);
    expectArraysClose(await gradients.data(), [11, 22, 33, 44]);
    expect(gradients.shape).toEqual([2, 2, 1]);
  });

  it('4D (tile) gradient', async () => {
    const x = tf.tensor4d([[[[1]], [[2]]], [[[3]], [[4]]]], [2, 2, 1, 1]);
    const dy = tf.tensor4d(
        [
          [[[.01, .1], [1, 10]], [[.02, .2], [2, 20]]],
          [[[.03, .3], [3, 30]], [[.04, .4], [4, 40]]]
        ],
        [2, 2, 2, 2]);
    const gradients = tf.grad(x => tf.tile(x, [1, 1, 2, 2]))(x, dy);
    expectArraysClose(await gradients.data(), [11.11, 22.22, 33.33, 44.44]);
    expect(gradients.shape).toEqual([2, 2, 1, 1]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.tile({} as tf.Tensor, [1]))
        .toThrowError(/Argument 'x' passed to 'tile' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const res = tf.tile([1, 2, 3], [2]);
    expect(res.shape).toEqual([6]);
    expectArraysClose(await res.data(), [1, 2, 3, 1, 2, 3]);
  });
});
