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

let defaultValue: tf.Scalar;
describeWithFlags('sparseToDense', ALL_ENVS, () => {
  beforeEach(() => defaultValue = tf.scalar(0, 'int32'));
  it('should work for scalar indices', async () => {
    const indices = tf.scalar(2, 'int32');
    const values = tf.scalar(100, 'int32');
    const shape = [6];
    const result = tf.sparseToDense(indices, values, shape, defaultValue);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(values.dtype);
    expectArraysClose(await result.data(), [0, 0, 100, 0, 0, 0]);
  });
  it('should work for vector', async () => {
    const indices = tf.tensor1d([0, 2, 4], 'int32');
    const values = tf.tensor1d([100, 101, 102], 'int32');
    const shape = [6];
    const result = tf.sparseToDense(indices, values, shape, defaultValue);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(values.dtype);
    expectArraysClose(await result.data(), [100, 0, 101, 0, 102, 0]);
  });
  it('should work for scalar value', async () => {
    const indices = tf.tensor1d([0, 2, 4], 'int32');
    const values = tf.scalar(10, 'int32');
    const shape = [6];
    const result = tf.sparseToDense(indices, values, shape, defaultValue);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(values.dtype);
    expectArraysClose(await result.data(), [10, 0, 10, 0, 10, 0]);
  });
  it('should work for matrix', async () => {
    const indices = tf.tensor2d([0, 1, 1, 1], [2, 2], 'int32');
    const values = tf.tensor1d([5, 6], 'float32');
    const shape = [2, 2];
    const result =
        tf.sparseToDense(indices, values, shape, defaultValue.toFloat());
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(values.dtype);
    expectArraysClose(await result.data(), [0, 5, 0, 6]);
  });

  it('should throw exception if default value does not match dtype', () => {
    const indices = tf.tensor2d([0, 1, 1, 1], [2, 2], 'int32');
    const values = tf.tensor1d([5, 6], 'float32');
    const shape = [2, 2];
    expect(
        () => tf.sparseToDense(indices, values, shape, tf.scalar(1, 'int32')))
        .toThrowError();
  });

  it('should allow setting default value', async () => {
    const indices = tf.tensor2d([0, 1, 1, 1], [2, 2], 'int32');
    const values = tf.tensor1d([5, 6], 'float32');
    const shape = [2, 2];
    const result = tf.sparseToDense(indices, values, shape, tf.scalar(1));
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(values.dtype);
    expectArraysClose(await result.data(), [1, 5, 1, 6]);
  });

  it('no default value passed', async () => {
    const indices = tf.tensor2d([0, 1, 1, 1], [2, 2], 'int32');
    const values = tf.tensor1d([5, 6], 'float32');
    const shape = [2, 2];
    const result = tf.sparseToDense(indices, values, shape);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(values.dtype);
    expectArraysClose(await result.data(), [0, 5, 0, 6]);
  });

  it('should support TensorLike inputs', async () => {
    const indices = [[0, 1], [1, 1]];
    const values = [5, 6];
    const shape = [2, 2];
    const result =
        tf.sparseToDense(indices, values, shape, defaultValue.toFloat());
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual('float32');
    expectArraysClose(await result.data(), [0, 5, 0, 6]);
  });

  it('should work with 0-sized tensors', async () => {
    const indices = tf.zeros([0], 'int32');
    const values = tf.zeros([0]);
    const defaultValue = tf.scalar(5);
    const result = tf.sparseToDense(indices, values, [3], defaultValue);
    expectArraysClose(await result.data(), [5, 5, 5]);
  });

  it('should throw error when indices are not int32', () => {
    const indices = tf.scalar(2, 'float32');
    const values = tf.scalar(100, 'int32');
    const shape = [6];
    expect(() => tf.sparseToDense(indices, values, shape, defaultValue))
        .toThrow();
  });

  it('should throw error when indices rank > 2', () => {
    const indices = tf.tensor3d([1], [1, 1, 1], 'int32');
    const values = tf.tensor1d([100], 'float32');
    const shape = [6];
    expect(() => tf.sparseToDense(indices, values, shape, defaultValue))
        .toThrow();
  });

  it('should throw error when values has rank > 1', () => {
    const indices = tf.tensor1d([0, 4, 2], 'int32');
    const values = tf.tensor2d([1.0, 2.0, 3.0], [3, 1], 'float32');
    const shape = [6];
    expect(() => tf.sparseToDense(indices, values, shape, defaultValue))
        .toThrow();
  });

  it('should throw error when values has wrong size', () => {
    const indices = tf.tensor1d([0, 4, 2], 'int32');
    const values = tf.tensor1d([1.0, 2.0, 3.0, 4.0], 'float32');
    const shape = [6];
    expect(() => tf.sparseToDense(indices, values, shape, defaultValue))
        .toThrow();
  });
});
