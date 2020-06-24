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

describeWithFlags('scatterND', ALL_ENVS, () => {
  it('should work for 2d', async () => {
    const indices = tf.tensor1d([0, 4, 2], 'int32');
    const updates = tf.tensor2d(
        [100, 101, 102, 777, 778, 779, 1000, 1001, 1002], [3, 3], 'int32');
    const shape = [5, 3];
    const result = tf.scatterND(indices, updates, shape);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(updates.dtype);
    expectArraysClose(
        await result.data(),
        [100, 101, 102, 0, 0, 0, 1000, 1001, 1002, 0, 0, 0, 777, 778, 779]);
  });

  it('should work for simple 1d', async () => {
    const indices = tf.tensor1d([3], 'int32');
    const updates = tf.tensor1d([101], 'float32');
    const shape = [5];
    const result = tf.scatterND(indices, updates, shape);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(updates.dtype);
    expectArraysClose(await result.data(), [0, 0, 0, 101, 0]);
  });

  it('should work for multiple 1d', async () => {
    const indices = tf.tensor1d([0, 4, 2], 'int32');
    const updates = tf.tensor1d([100, 101, 102], 'float32');
    const shape = [5];
    const result = tf.scatterND(indices, updates, shape);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(updates.dtype);
    expectArraysClose(await result.data(), [100, 0, 102, 0, 101]);
  });

  it('should work for high rank updates', async () => {
    const indices = tf.tensor2d([0, 2], [2, 1], 'int32');
    const updates = tf.tensor3d(
        [
          5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
          5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8
        ],
        [2, 4, 4], 'float32');
    const shape = [4, 4, 4];
    const result = tf.scatterND(indices, updates, shape);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(updates.dtype);
    expectArraysClose(await result.data(), [
      5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
      8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
  });

  it('should work for high rank indices', async () => {
    const indices = tf.tensor2d([0, 2, 0, 1], [2, 2], 'int32');
    const updates = tf.tensor1d([10, 20], 'float32');
    const shape = [3, 3];
    const result = tf.scatterND(indices, updates, shape);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(updates.dtype);
    expectArraysClose(await result.data(), [0, 20, 10, 0, 0, 0, 0, 0, 0]);
  });

  it('should work for high rank indices and update', () => {
    const indices = tf.tensor2d([1, 0, 0, 1, 0, 1], [3, 2], 'int32');
    const updates = tf.ones([3, 256], 'float32');
    const shape = [2, 2, 256];
    const result = tf.scatterND(indices, updates, shape);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(updates.dtype);
  });

  it('should sum the duplicated indices', async () => {
    const indices = tf.tensor1d([0, 4, 2, 1, 3, 0], 'int32');
    const updates = tf.tensor1d([10, 20, 30, 40, 50, 60], 'float32');
    const shape = [8];
    const result = tf.scatterND(indices, updates, shape);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(updates.dtype);
    expectArraysClose(await result.data(), [70, 40, 30, 50, 20, 0, 0, 0]);
  });

  it('should work for tensorLike input', async () => {
    const indices = [0, 4, 2];
    const updates = [100, 101, 102];
    const shape = [5];
    const result = tf.scatterND(indices, updates, shape);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual('float32');
    expectArraysClose(await result.data(), [100, 0, 102, 0, 101]);
  });

  it('should throw error when indices type is not int32', () => {
    const indices = tf.tensor2d([0, 2, 0, 1], [2, 2], 'float32');
    const updates = tf.tensor1d([10, 20], 'float32');
    const shape = [3, 3];
    expect(() => tf.scatterND(indices, updates, shape)).toThrow();
  });

  it('should throw error when indices and update mismatch', () => {
    const indices = tf.tensor2d([0, 4, 2], [3, 1], 'int32');
    const updates = tf.tensor2d(
        [100, 101, 102, 103, 777, 778, 779, 780, 10000, 10001, 10002, 10004],
        [3, 4], 'float32');
    const shape = [5, 3];
    expect(() => tf.scatterND(indices, updates, shape)).toThrow();
  });

  it('should throw error when indices and update count mismatch', () => {
    const indices = tf.tensor2d([0, 4, 2], [3, 1], 'int32');
    const updates =
        tf.tensor2d([100, 101, 102, 10000, 10001, 10002], [2, 3], 'float32');
    const shape = [5, 3];
    expect(() => tf.scatterND(indices, updates, shape)).toThrow();
  });

  it('should throw error when indices are scalar', () => {
    const indices = tf.scalar(1, 'int32');
    const updates =
        tf.tensor2d([100, 101, 102, 10000, 10001, 10002], [2, 3], 'float32');
    const shape = [5, 3];
    expect(() => tf.scatterND(indices, updates, shape)).toThrow();
  });

  it('should throw error when update is scalar', () => {
    const indices = tf.tensor2d([0, 4, 2], [3, 1], 'int32');
    const updates = tf.scalar(1, 'float32');
    const shape = [5, 3];
    expect(() => tf.scatterND(indices, updates, shape)).toThrow();
  });
});
