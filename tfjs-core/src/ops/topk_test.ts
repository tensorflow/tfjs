/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {scalar} from './scalar';
import {tensor1d} from './tensor1d';
import {tensor2d} from './tensor2d';
import {tensor3d} from './tensor3d';

describeWithFlags('topk', ALL_ENVS, () => {
  beforeAll(() => {
    // Ensure WebGL environment uses GPU
    if (tf.getBackend() === 'webgl') {
      tf.env().set('TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD', 0);
      tf.env().set('TOPK_K_CPU_HANDOFF_THRESHOLD', 1024);
    }
  });

  it('1d array with k = 0', async () => {
    const a = tensor1d([20, 10, 40, 30]);
    const {values, indices} = tf.topk(a, 0);

    expect(values.shape).toEqual([0]);
    expect(indices.shape).toEqual([0]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(await values.data(), []);
    expectArraysClose(await indices.data(), []);
  });

  it('1d array with length 1', async () => {
    const a = tensor1d([20]);
    const {values, indices} = tf.topk(a, 1);

    expect(values.shape).toEqual([1]);
    expect(indices.shape).toEqual([1]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(await values.data(), [20]);
    expectArraysClose(await indices.data(), [0]);
  });

  it('1d array with default k', async () => {
    const a = tensor1d([20, 10, 40, 30]);
    const {values, indices} = tf.topk(a);

    expect(values.shape).toEqual([1]);
    expect(indices.shape).toEqual([1]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(await values.data(), [40]);
    expectArraysClose(await indices.data(), [2]);
  });

  it('1d array with default k from tensor.topk', async () => {
    const a = tensor1d([20, 10, 40, 30]);
    const {values, indices} = a.topk();

    expect(values.shape).toEqual([1]);
    expect(indices.shape).toEqual([1]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(await values.data(), [40]);
    expectArraysClose(await indices.data(), [2]);
  });

  it('2d array with default k', async () => {
    const a = tensor2d([[10, 50], [40, 30]]);
    const {values, indices} = tf.topk(a);

    expect(values.shape).toEqual([2, 1]);
    expect(indices.shape).toEqual([2, 1]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(await values.data(), [50, 40]);
    expectArraysClose(await indices.data(), [1, 0]);
  });

  it('2d array with k=2', async () => {
    const a = tensor2d([
      [1, 5, 2],
      [4, 3, 6],
      [3, 2, 1],
      [1, 2, 3],
    ]);
    const k = 2;
    const {values, indices} = tf.topk(a, k);

    expect(values.shape).toEqual([4, 2]);
    expect(indices.shape).toEqual([4, 2]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(await values.data(), [5, 2, 6, 4, 3, 2, 3, 2]);
    expectArraysClose(await indices.data(), [1, 2, 2, 0, 0, 1, 2, 1]);
  });

  it('2d array with k=2 from tensor.topk', async () => {
    const a = tensor2d([
      [1, 5, 2],
      [4, 3, 6],
      [3, 2, 1],
      [1, 2, 3],
    ]);
    const k = 2;
    const {values, indices} = a.topk(k);

    expect(values.shape).toEqual([4, 2]);
    expect(indices.shape).toEqual([4, 2]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(await values.data(), [5, 2, 6, 4, 3, 2, 3, 2]);
    expectArraysClose(await indices.data(), [1, 2, 2, 0, 0, 1, 2, 1]);
  });

  it('3d array with k=3', async () => {
    const a = tensor3d([
      [[1, 5, 2], [4, 3, 6]],
      [[3, 2, 1], [1, 2, 3]],
    ]);  // 2x2x3.
    const k = 3;
    const {values, indices} = tf.topk(a, k);

    expect(values.shape).toEqual([2, 2, 3]);
    expect(indices.shape).toEqual([2, 2, 3]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(
        await values.data(), [5, 2, 1, 6, 4, 3, 3, 2, 1, 3, 2, 1]);
    expectArraysClose(
        await indices.data(), [1, 2, 0, 2, 0, 1, 0, 1, 2, 2, 1, 0]);
  });

  it('topk(int32) propagates int32 dtype', async () => {
    const a = tensor1d([2, 3, 1, 4], 'int32');
    const {values, indices} = tf.topk(a);

    expect(values.shape).toEqual([1]);
    expect(indices.shape).toEqual([1]);
    expect(values.dtype).toBe('int32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(await values.data(), [4]);
    expectArraysClose(await indices.data(), [3]);
  });

  it('lower-index element appears first, k=4', async () => {
    const a = tensor1d([1, 2, 2, 1], 'int32');
    const k = 4;
    const {values, indices} = tf.topk(a, k);

    expect(values.shape).toEqual([4]);
    expect(indices.shape).toEqual([4]);
    expect(values.dtype).toBe('int32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(await values.data(), [2, 2, 1, 1]);
    expectArraysClose(await indices.data(), [1, 2, 0, 3]);
  });

  it('lower-index element appears first, k=65', async () => {
    const a = [
      1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2,
      1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2,
      1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1
    ];
    const k = a.length;
    const {values, indices} = tf.topk(a, k);

    expectArraysClose(await values.data(), [
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]);
    expectArraysClose(await indices.data(), [
      2,  4,  8,  9,  12, 15, 18, 21, 23, 27, 30, 33, 38, 40, 41, 42, 43,
      45, 47, 48, 49, 51, 52, 54, 55, 57, 61, 63, 0,  1,  3,  5,  6,  7,
      10, 11, 13, 14, 16, 17, 19, 20, 22, 24, 25, 26, 28, 29, 31, 32, 34,
      35, 36, 37, 39, 44, 46, 50, 53, 56, 58, 59, 60, 62, 64
    ]);
  });

  it('lower-index element appears first, sorted=false', async () => {
    const a = [
      1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2,
      1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2,
      1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1
    ];
    const k = a.length;
    const {values, indices} = tf.topk(a, k, false);

    expect(values.shape).toEqual([k]);
    expect(indices.shape).toEqual([k]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');

    const valuesData = await values.data();
    valuesData.sort();
    expectArraysClose(valuesData, [
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    ]);

    const indicesData = await indices.data();
    const onesIndices = indicesData.filter((index: number) => a[index] === 1);
    const twosIndices = indicesData.filter((index: number) => a[index] === 2);
    expectArraysClose(onesIndices, [
      0,  1,  3,  5,  6,  7,  10, 11, 13, 14, 16, 17, 19,
      20, 22, 24, 25, 26, 28, 29, 31, 32, 34, 35, 36, 37,
      39, 44, 46, 50, 53, 56, 58, 59, 60, 62, 64
    ]);
    expectArraysClose(twosIndices, [
      2,  4,  8,  9,  12, 15, 18, 21, 23, 27, 30, 33, 38, 40,
      41, 42, 43, 45, 47, 48, 49, 51, 52, 54, 55, 57, 61, 63
    ]);
  });

  it('throws when k < 0', () => {
    const a = tensor2d([[10, 50], [40, 30]]);
    expect(() => tf.topk(a, -1))
        .toThrowError(/'k' passed to topk\(\) must be >= 0/);
  });

  it('throws when k > size of array', () => {
    const a = tensor2d([[10, 50], [40, 30]]);
    expect(() => tf.topk(a, 3))
        .toThrowError(/'k' passed to topk\(\) must be <= the last dimension/);
  });

  it('throws when passed a scalar', () => {
    const a = scalar(2);
    expect(() => tf.topk(a))
        .toThrowError(/topk\(\) expects the input to be of rank 1 or higher/);
  });

  it('negative infinity input', async () => {
    const a = [-Infinity, -Infinity, -Infinity, -Infinity, -Infinity];
    const k = a.length;
    const {values, indices} = tf.topk(a, k);

    expect(values.shape).toEqual([k]);
    expect(indices.shape).toEqual([k]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(await values.data(), a);
    expectArraysClose(await indices.data(), [0, 1, 2, 3, 4]);
  });

  it('accepts a tensor-like object, k=2', async () => {
    const a = [20, 10, 40, 30];
    const k = 2;
    const {values, indices} = tf.topk(a, k);

    expect(values.shape).toEqual([2]);
    expect(indices.shape).toEqual([2]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(await values.data(), [40, 30]);
    expectArraysClose(await indices.data(), [2, 3]);
  });

  it('handles output tensors from other ops', async () => {
    const a = tensor1d([20, 10, 40, 30]);
    const b = scalar(2);
    const {values, indices} = tf.topk(tf.add(a, b));

    expect(values.shape).toEqual([1]);
    expect(indices.shape).toEqual([1]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expectArraysClose(await values.data(), [42]);
    expectArraysClose(await indices.data(), [2]);
  });
});
