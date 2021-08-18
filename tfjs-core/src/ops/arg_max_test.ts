/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

import * as reduce_util from './reduce_util';

describeWithFlags('argmax', ALL_ENVS, () => {
  it('Tensor1D', async () => {
    const a = tf.tensor1d([1, 0, 3, 2]);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), 2);
  });

  it('one value', async () => {
    const a = tf.tensor1d([10]);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), 0);
  });

  it('N > than parallelization threshold', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = i;
    }
    const a = tf.tensor1d(values);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), n - 1);
  });

  it('3D, N > than parallelization threshold', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = i;
    }
    const a = tf.tensor3d(values, [1, 1, n]);
    const result = tf.argMax(a, -1);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), n - 1);
  });

  it('max index corresponds to start of a non-initial window', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const windowSize = reduce_util.computeOptimalWindowSize(n);
    const values = new Float32Array(n);
    const index = windowSize * 2;
    values[index] = 1;
    const a = tf.tensor1d(values);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), index);
  });

  it('5D, max index corresponds to start of a non-initial window', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const windowSize = reduce_util.computeOptimalWindowSize(n);
    const values = new Float32Array(n);
    const index = windowSize * 2;
    values[index] = 1;
    const a = tf.tensor5d(values, [1, 1, 1, 1, n]);
    const result = tf.argMax(a, -1);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), index);
  });

  it('ignores NaNs', async () => {
    const a = tf.tensor1d([0, 3, 5, NaN, 3]);
    const res = tf.argMax(a);
    expect(res.dtype).toBe('int32');
    expectArraysEqual(await res.data(), 2);
  });

  it('2D, no axis specified', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysEqual(await tf.argMax(a).data(), [1, 0, 1]);
  });

  it('4D, no axis specified', async () => {
    const a = tf.tensor4d([3, -1, 0, 100, -7, 2], [2, 1, 1, 3]);
    expectArraysEqual(await tf.argMax(a).data(), [1, 0, 1]);
  });

  it('2D, axis=0', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, 0);

    expect(r.shape).toEqual([3]);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [1, 0, 1]);
  });

  it('6D, axis=0', async () => {
    const a = tf.tensor6d([3, -1, 0, 100, -7, 2], [2, 1, 1, 1, 1, 3]);
    const r = tf.argMax(a, 0);

    expect(r.shape).toEqual([1, 1, 1, 1, 3]);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [1, 0, 1]);
  });

  it('2D, axis=1', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, 1);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [2, 0]);
  });

  it('2D, axis = -1', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, -1);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [2, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.argMax({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'argMax' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.argMax([1, 0, 3, 2]);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), 2);
  });

  it('accepts tensor with bool values', async () => {
    const t = tf.tensor1d([0, 1], 'bool');
    const result = tf.argMax(t);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), 1);
  });

  it('has gradient', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const dy = tf.ones([3], 'float32');
    const da = tf.grad((x: tf.Tensor2D) => tf.argMax(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const dy = tf.ones([3], 'float32');
    const da = tf.grad((x: tf.Tensor2D) => tf.argMax(x.clone()).clone())(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.argMax(['a']))
        .toThrowError(/Argument 'x' passed to 'argMax' must be numeric tensor/);
  });
});
