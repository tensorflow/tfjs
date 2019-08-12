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

import {scalar, tensor1d, tensor2d, tensor3d} from './tensor_ops';

describeWithFlags('topk', ALL_ENVS, () => {
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
});
