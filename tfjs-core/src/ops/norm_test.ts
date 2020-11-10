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

describeWithFlags('norm', ALL_ENVS, () => {
  it('scalar norm', async () => {
    const a = tf.scalar(-22.0);
    const norm = tf.norm(a);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 22);
  });

  it('vector inf norm', async () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, Infinity);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 4);
  });

  it('vector -inf norm', async () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, -Infinity);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 1);
  });

  it('vector 1 norm', async () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, 1);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 10);
  });

  it('vector euclidean norm', async () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, 'euclidean');

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 5.4772);
  });

  it('vector 2-norm', async () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, 2);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 5.4772);
  });

  it('vector >2-norm to throw error', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    expect(() => tf.norm(a, 3)).toThrowError();
  });

  it('matrix inf norm', async () => {
    const a = tf.tensor2d([1, 2, -3, 1, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 4);
  });

  it('matrix -inf norm', async () => {
    const a = tf.tensor2d([1, 2, -3, 1, 0, 1], [3, 2]);
    const norm = tf.norm(a, -Infinity, [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 1);
  });

  it('matrix 1 norm', async () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    const norm = tf.norm(a, 1, [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 5);
  });

  it('matrix euclidean norm', async () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    const norm = tf.norm(a, 'euclidean', [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 4.123);
  });

  it('matrix fro norm', async () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    const norm = tf.norm(a, 'fro', [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 4.123);
  });

  it('matrix other norm to throw error', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    expect(() => tf.norm(a, 2, [0, 1])).toThrowError();
  });

  it('propagates NaNs for norm', async () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    const norm = tf.norm(a);

    expect(norm.dtype).toBe('float32');
    expectArraysEqual(await norm.data(), NaN);
  });

  it('axis=null in 2D array norm', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity);

    expect(norm.shape).toEqual([]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('2D array norm with keep dim', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, null, true /* keepDims */);

    expect(norm.shape).toEqual([1, 1]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('axis=0 in 2D array norm', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0]);

    expect(norm.shape).toEqual([2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3, 2]);
  });

  it('axis=1 in 2D array norm', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [1]);

    expect(norm.dtype).toBe('float32');
    expect(norm.shape).toEqual([3]);
    expectArraysClose(await norm.data(), [2, 3, 1]);
  });

  it('axis=1 keepDims in 2D array norm', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [1], true);

    expect(norm.dtype).toBe('float32');
    expect(norm.shape).toEqual([3, 1]);
    expectArraysClose(await norm.data(), [2, 3, 1]);
  });

  it('2D norm with axis=1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const norm = tf.norm(a, Infinity, 1);

    expect(norm.shape).toEqual([2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3, 1]);
  });

  it('axis=0,1 in 2D array norm', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.shape).toEqual([]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('axis=0,1 keepDims in 2D array norm', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0, 1], true);

    expect(norm.shape).toEqual([1, 1]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('3D norm axis=0,1, matrix inf norm', async () => {
    const a = tf.tensor3d([1, 2, -3, 1, 0, 1], [3, 2, 1]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.shape).toEqual([1]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [4]);
  });

  it('axis=0,1 keepDims in 3D array norm', async () => {
    const a = tf.tensor3d([1, 2, 3, 0, 0, 1], [3, 2, 1]);
    const norm = tf.norm(a, Infinity, [0, 1], true);

    expect(norm.shape).toEqual([1, 1, 1]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('axis=0,1 keepDims in 3D array norm', async () => {
    const a = tf.tensor3d([1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1], [3, 2, 2]);
    const norm = tf.norm(a, Infinity, [0, 1], true);

    expect(norm.shape).toEqual([1, 1, 2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [4, 3]);
  });

  it('axis=null in 3D array norm', async () => {
    const a = tf.tensor3d([1, 2, 3, 0, 0, 1], [3, 2, 1]);
    const norm = tf.norm(a, Infinity);

    expect(norm.shape).toEqual([]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('axis=null in 4D array norm', async () => {
    const a = tf.tensor4d([1, 2, 3, 0, 0, 1], [3, 2, 1, 1]);
    const norm = tf.norm(a, Infinity);

    expect(norm.shape).toEqual([]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('axis=0,1 in 4D array norm', async () => {
    const a = tf.tensor4d(
        [
          1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1,
          1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1
        ],
        [3, 2, 2, 2]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.shape).toEqual([2, 2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [4, 3, 4, 3]);
  });

  it('axis=0,1 in 4D array norm', async () => {
    const a = tf.tensor4d(
        [
          1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1,
          1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1
        ],
        [3, 2, 2, 2]);
    const norm = tf.norm(a, Infinity, [0, 1], true);

    expect(norm.shape).toEqual([1, 1, 2, 2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [4, 3, 4, 3]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.norm({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'norm' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const norm = tf.norm([1, -2, 3, -4], 1);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 10);
  });

  it('throws error for string tensors', () => {
    expect(() => tf.norm([
      'a', 'b'
    ])).toThrowError(/Argument 'x' passed to 'norm' must be numeric tensor/);
  });
});
