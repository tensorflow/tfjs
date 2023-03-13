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
import {Rank} from '../types';

describeWithFlags('slice2d', ALL_ENVS, () => {
  it('slicing a 1x1 from a 1x1 returns a 1x1', () => {
    const a = tf.tensor2d([0], [1, 1]);
    const b = tf.slice2d(a, [0, 0], [1, 1]);
    expect(b.shape).toEqual([1, 1]);
  });

  it('returns a tensor of slice size', () => {
    const a = tf.zeros<Rank.R2>([100, 100]);
    const b = tf.slice2d(a, [0, 0], [12, 34]);
    expect(b.shape).toEqual([12, 34]);
  });

  it('returns the upper-left submatrix when begin is [0, 0]', async () => {
    const a = tf.randomUniform<Rank.R2>([10, 10], -1, 1);
    const b = tf.slice2d(a, [0, 0], [2, 2]);
    const aValues = await a.data();

    expectArraysClose(
        await b.data(), [aValues[0], aValues[1], aValues[10], aValues[11]]);
  });

  it('returns the rectangle specified', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3]);
    const b = tf.slice2d(a, [1, 1], [3, 2]);

    expectArraysClose(await b.data(), [5, 6, 8, 9, 11, 12]);
  });

  it('throws when requesting out of bounds slice', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3]);
    expect(() => tf.slice2d(a, [1, 1], [10, 10])).toThrowError();
  });

  it('grad', async () => {
    const a = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
    const dy = tf.tensor2d([[20], [50]]);
    const da =
        tf.grad((x: tf.Tensor2D) => tf.slice2d(a, [0, 1], [2, 1]))(a, dy);
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [0, 20, 0, 0, 50, 0]);
  });

  it('accepts a tensor-like object', () => {
    const a = [[0]];  // 1x1
    const b = tf.slice2d(a, [0, 0], [1, 1]);
    expect(b.shape).toEqual([1, 1]);
  });

  it('slice an already sliced tensor, first was not continous', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [0, 1]);
    const c = tf.slice(b, [1, 1], [1, 1]);
    expect(c.shape).toEqual([1, 1]);
    expectArraysClose(await c.data(), [7]);
  });

  it('slice an already sliced tensor, first was continous', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [1, 0]);
    const c = tf.slice(b, [1, 0]);
    expect(c.shape).toEqual([1, 4]);
    expectArraysClose(await c.data(), [9, 10, 11, 12]);
  });

  it('slice an already sliced tensor and do async read', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [0, 1]);
    const c = tf.slice(b, [1, 1], [1, 1]);
    expect(c.shape).toEqual([1, 1]);
    expectArraysClose(await c.data(), new Float32Array([7]));
  });

  it('square a sliced texture, followed by non-sliced texture of same shape',
     async () => {  // Tests collisions in the shader cache.
       // Make a 2x3 tensor, upload to gpu and reshape to 3x2.
       const input = tf.tensor([[1, 2, 3], [4, 5, 6]]).abs().as2D(3, 2);
       const slicedInput = tf.slice(input, [0, 0], [3, 2]);
       // First square program takes the sliced input.
       const a = slicedInput.square();
       expectArraysClose(await a.data(), [1, 4, 9, 16, 25, 36]);
       // Second square program takes the non-sliced input.
       const b = tf.square(input);
       expectArraysClose(await b.data(), [1, 4, 9, 16, 25, 36]);
     });

  it('square a non-sliced texture, followed by a sliced texture of same shape',
     async () => {  // Tests collisions in the shader cache.
       // Make a 2x3 tensor, upload to gpu and reshape to 3x2.
       const input = tf.tensor([[1, 2, 3], [4, 5, 6]]).abs().as2D(3, 2);
       // Make a sliced version of the same tensor with the same shape.
       const slicedInput = tf.slice(input, [0, 0], [3, 2]);
       // First square program takes the non-sliced input.
       const a = input.square();
       expectArraysClose(await a.data(), [1, 4, 9, 16, 25, 36]);
       // Second square program takes the sliced input.
       const b = tf.square(slicedInput);
       expectArraysClose(await b.data(), [1, 4, 9, 16, 25, 36]);
     });

  it('slice a tensor and do async read', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [0, 1], [3, 2]);
    expect(b.shape).toEqual([3, 2]);
    const vals = await b.data();
    expectArraysClose(vals, new Float32Array([2, 3, 6, 7, 10, 11]));
  });

  it('flatten a sliced tensor that was continuous in memory', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [1, 0]).flatten();
    expect(b.shape).toEqual([8]);
    expectArraysClose(await b.data(), [5, 6, 7, 8, 9, 10, 11, 12]);
  });

  it('slice a tensor that was not continuous in memory', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [0, 1]);
    expect(b.shape).toEqual([3, 3]);
    expectArraysClose(await b.data(), [2, 3, 4, 6, 7, 8, 10, 11, 12]);
  });

  it('flatten a sliced tensor that was not continuous in memory', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [0, 1]).flatten();
    expect(b.shape).toEqual([9]);
    expectArraysClose(await b.data(), [2, 3, 4, 6, 7, 8, 10, 11, 12]);
  });

  it('flatten a sliced tensor not continuous in memory and run program',
     async () => {
       const a = [
         [1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
       ];  // 3x4.
       const b = tf.slice(a, [0, 1]).flatten();
       const c = tf.square(b);
       expectArraysClose(await c.data(), [4, 9, 16, 36, 49, 64, 100, 121, 144]);
     });

  it('reshape a sliced 1d into a 2d tensor', async () => {
    const a = [1, 2, 3, 4, 5];
    const b = tf.slice(a, 1).as2D(2, 2);
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(await b.data(), [2, 3, 4, 5]);
  });

  it('reshape a sliced 1d into a 2d tensor and run program', async () => {
    const a = [1, 2, 3, 4, 5];
    const b = tf.slice(a, 1).as2D(2, 2).square();
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(await b.data(), [4, 9, 16, 25]);
  });

  it('broadcast the original with the sliced tensor', async () => {
    const a = [[1, 2], [3, 4]];
    const b = tf.slice(a, [0, 1]);
    const c = tf.add(a, b);
    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(await c.data(), [3, 4, 7, 8]);
  });

  it('zero-sized slice out of a non-zero sized tensor', async () => {
    const a = tf.zeros([4, 2]);
    const res = tf.slice(a, [0, 0], [0, 2]);
    expect(res.shape).toEqual([0, 2]);
    expectArraysClose(await res.data(), []);
  });

  it('zero-sized slice out of a zero-sized tensor', async () => {
    const a = tf.zeros([0, 4]);
    const res = tf.slice(a, [0, 1], [0, 3]);
    expect(res.shape).toEqual([0, 3]);
    expectArraysClose(await res.data(), []);
  });
});
