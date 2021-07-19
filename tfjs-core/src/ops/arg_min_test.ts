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

describeWithFlags('argmin', ALL_ENVS, () => {
  it('Tensor1D', async () => {
    const a = tf.tensor1d([1, 0, 3, 2]);
    const result = tf.argMin(a);
    expectArraysEqual(await result.data(), 1);
  });

  it('one value', async () => {
    const a = tf.tensor1d([10]);
    const result = tf.argMin(a);
    expectArraysEqual(await result.data(), 0);
  });

  it('N > than parallelization threshold', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = n - i;
    }
    const a = tf.tensor1d(values);
    const result = tf.argMin(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), n - 1);
  });

  it('4D, N > than parallelization threshold', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = n - i;
    }
    const a = tf.tensor4d(values, [1, 1, 1, n]);
    const result = tf.argMin(a, -1);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), n - 1);
  });

  it('min index corresponds to start of a non-initial window', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const windowSize = reduce_util.computeOptimalWindowSize(n);
    const values = new Float32Array(n);
    const index = windowSize * 2;
    values[index] = -1;
    const a = tf.tensor1d(values);
    const result = tf.argMin(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), index);
  });

  it('ignores NaNs', async () => {
    const a = tf.tensor1d([5, 0, NaN, -1, 3]);
    const res = tf.argMin(a);
    expectArraysEqual(await res.data(), 3);
  });

  it('3D, ignores NaNs', async () => {
    const a = tf.tensor3d([5, 0, NaN, -1, 3], [1, 1, 5]);
    const res = tf.argMin(a, -1);
    expectArraysEqual(await res.data(), 3);
  });

  it('2D, no axis specified', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysEqual(await tf.argMin(a).data(), [0, 1, 0]);
  });

  it('2D, axis=0', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.argMin(a, 0);

    expect(r.shape).toEqual([3]);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [0, 1, 0]);
  });

  it('2D, axis=1', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, -8], [2, 3]);
    const r = tf.argMin(a, 1);
    expectArraysEqual(await r.data(), [1, 2]);
  });

  it('2D, axis = -1', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, -8], [2, 3]);
    const r = tf.argMin(a, -1);
    expectArraysEqual(await r.data(), [1, 2]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.argMin({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'argMin' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.argMin([1, 0, 3, 2]);
    expectArraysEqual(await result.data(), 1);
  });

  it('accepts tensor with bool values', async () => {
    const t = tf.tensor1d([0, 1], 'bool');
    const result = tf.argMin(t);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), 0);
  });

  it('has gradient', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const dy = tf.ones([3], 'float32');
    const da = tf.grad((x: tf.Tensor2D) => tf.argMin(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const dy = tf.ones([3], 'float32');
    const da = tf.grad((x: tf.Tensor2D) => tf.argMin(x.clone()).clone())(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.argMin(['a']))
        .toThrowError(/Argument 'x' passed to 'argMin' must be numeric tensor/);
  });
});
