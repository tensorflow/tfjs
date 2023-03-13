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

describeWithFlags('setdiff1dAsync', ALL_ENVS, () => {
  it('1d int32 tensor', async () => {
    const x = tf.tensor1d([1, 2, 3, 4], 'int32');
    const y = tf.tensor1d([1, 2], 'int32');
    const [out, indices] = await tf.setdiff1dAsync(x, y);
    expect(out.dtype).toBe('int32');
    expect(indices.dtype).toBe('int32');
    expect(out.shape).toEqual([2]);
    expect(indices.shape).toEqual([2]);
    expectArraysClose(await out.data(), [3, 4]);
    expectArraysClose(await indices.data(), [2, 3]);
  });

  it('1d float32 tensor', async () => {
    const x = tf.tensor1d([1, 2, 3, 4], 'float32');
    const y = tf.tensor1d([1, 3], 'float32');
    const [out, indices] = await tf.setdiff1dAsync(x, y);
    expect(out.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expect(out.shape).toEqual([2]);
    expect(indices.shape).toEqual([2]);
    expectArraysClose(await out.data(), [2, 4]);
    expectArraysClose(await indices.data(), [1, 3]);
  });

  it('empty output', async () => {
    const x = tf.tensor1d([1, 2, 3, 4], 'float32');
    const y = tf.tensor1d([1, 2, 3, 4], 'float32');
    const [out, indices] = await tf.setdiff1dAsync(x, y);
    expect(out.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expect(out.shape).toEqual([0]);
    expect(indices.shape).toEqual([0]);
    expectArraysClose(await out.data(), []);
    expectArraysClose(await indices.data(), []);
  });

  it('tensor like', async () => {
    const x = [1, 2, 3, 4];
    const y = [1, 3];
    const [out, indices] = await tf.setdiff1dAsync(x, y);
    expect(out.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');
    expect(out.shape).toEqual([2]);
    expect(indices.shape).toEqual([2]);
    expectArraysClose(await out.data(), [2, 4]);
    expectArraysClose(await indices.data(), [1, 3]);
  });

  it('should throw if x is not 1d', async () => {
    const x = tf.tensor2d([1, 2, 3, 4], [4, 1], 'float32');
    const y = tf.tensor1d([1, 2, 3, 4], 'float32');
    try {
      await tf.setdiff1dAsync(x, y);
      throw new Error('The line above should have thrown an error');
    } catch (ex) {
      expect(ex.message).toBe('x should be 1D tensor, but got x (4,1).');
    }
  });

  it('should throw if y is not 1d', async () => {
    const x = tf.tensor1d([1, 2, 3, 4], 'float32');
    const y = tf.tensor2d([1, 2, 3, 4], [4, 1], 'float32');
    try {
      await tf.setdiff1dAsync(x, y);
      throw new Error('The line above should have thrown an error');
    } catch (ex) {
      expect(ex.message).toBe('y should be 1D tensor, but got y (4,1).');
    }
  });

  it('should throw if x and y dtype mismatch', async () => {
    const x = tf.tensor1d([1, 2, 3, 4], 'float32');
    const y = tf.tensor1d([1, 2, 3, 4], 'int32');
    try {
      await tf.setdiff1dAsync(x, y);
      throw new Error('The line above should have thrown an error');
    } catch (ex) {
      expect(ex.message)
          .toBe(
              'x and y should have the same dtype,' +
              ' but got x (float32) and y (int32).');
    }
  });
});
