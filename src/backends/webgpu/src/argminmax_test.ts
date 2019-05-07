/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';
import * as tfwebgpu from './index';

function computeArgMinMax(values: number[], reduceType: 'min'|'max') {
  return values.map((val, index) => ({val, index}))
      .reduce((acc, curr) => {
        const better =
            (reduceType === 'min' ? curr.val < acc.val : curr.val > acc.val);
        return better ? curr : acc;
      })
      .index;
}

function randomFloats(n: number) {
  const values = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    values[i] = i;
  }
  values.sort(() => Math.random());
  return values;
}

describe('Reduction: argmax', () => {
  beforeAll(async () => tfwebgpu.ready);

  it('Tensor1D', async () => {
    const a = tf.tensor1d([1, 0, 3, 2]);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(await result.data(), new Float32Array([2]));
  });

  it('one value', async () => {
    const a = tf.tensor1d([10]);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(await result.data(), new Float32Array([0]));
  });

  it('3D, axis=0', async () => {
    const values = [
      5,  8,  2, 11, 1, 6, 7, 10, 3, 9, 0,  4,   //
      0,  1,  2, 3,  4, 5, 6, 7,  8, 9, 10, 11,  //
      11, 10, 9, 8,  7, 6, 5, 4,  3, 2, 1,  0,   //
    ];
    const a = tf.tensor3d(values, [3, 4, 3]);
    const result = tf.argMax(a, 0);
    expect(result.shape).toEqual([4, 3]);
    expect(result.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(
        await result.data(),
        new Float32Array([2, 2, 2, 0, 2, 0, 0, 0, 1, 0, 1, 1]));
  });

  it('3D, axis=1', async () => {
    const values = [
      5,  8,  2,  //
      11, 1,  6,  //
      7,  10, 3,  //
      9,  0,  4,  //

      0,  1,  2,   //
      3,  4,  5,   //
      6,  7,  8,   //
      9,  10, 11,  //

      11, 10, 9,  //
      8,  7,  6,  //
      5,  4,  3,  //
      2,  1,  0,  //
    ];
    const a = tf.tensor3d(values, [3, 4, 3]);
    const result = tf.argMax(a, 1);
    expect(result.shape).toEqual([3, 3]);
    expect(result.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(
        await result.data(), new Float32Array([1, 2, 1, 3, 3, 3, 0, 0, 0]));
  });

  it('3D, axis=2', async () => {
    const values = [
      5,  8,  2, 11, 1, 6, 7, 10, 3, 9, 0,  4,   //
      0,  1,  2, 3,  4, 5, 6, 7,  8, 9, 10, 11,  //
      11, 10, 9, 8,  7, 6, 5, 4,  3, 2, 1,  0,   //
    ];
    const a = tf.tensor3d(values, [3, 4, 3]);
    const result = tf.argMax(a, 2);
    expect(result.shape).toEqual([3, 4]);
    expect(result.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(
        await result.data(),
        new Float32Array([1, 0, 1, 0, 2, 2, 2, 2, 0, 0, 0, 0]));
  });

  it('1D large', async () => {
    const n = 250000;
    const values = randomFloats(n);
    const argmax = computeArgMinMax(Array.from(values), 'max');

    const a = tf.tensor1d(values);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(
        await result.data(), new Float32Array([argmax]));
  });

  it('3D large, axis = -1', async () => {
    const n = 250000;
    const values = randomFloats(n);
    const argmax = computeArgMinMax(Array.from(values), 'max');

    const a = tf.tensor3d(values, [1, 1, n]);
    const result = tf.argMax(a, -1);
    expect(result.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(
        await result.data(), new Float32Array([argmax]));
  });

  it('ignores NaNs', async () => {
    const a = tf.tensor1d([0, 3, 5, NaN, 3]);
    const res = tf.argMax(a);
    expect(res.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(await res.data(), new Float32Array([2]));
  });

  it('2D, no axis specified', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    tf.test_util.expectArraysEqual(
        await tf.argMax(a).data(), new Float32Array([1, 0, 1]));
  });

  it('4D, no axis specified', async () => {
    const a = tf.tensor4d([3, -1, 0, 100, -7, 2], [2, 1, 1, 3]);
    tf.test_util.expectArraysEqual(
        await tf.argMax(a).data(), new Float32Array([1, 0, 1]));
  });

  it('2D, axis=0', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, 0);

    expect(r.shape).toEqual([3]);
    expect(r.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(await r.data(), new Float32Array([1, 0, 1]));
  });

  it('2D, axis=1', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, 1);
    expect(r.shape).toEqual([2]);
    expect(r.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(await r.data(), new Float32Array([2, 0]));
  });

  it('2D, axis = -1', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, -1);
    expect(r.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(await r.data(), new Float32Array([2, 0]));
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.argMax({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'argMax' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.argMax([1, 0, 3, 2]);
    expect(result.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(await result.data(), new Float32Array([2]));
  });

  it('throws error for string tensor', () => {
    expect(() => tf.argMax(['a']))
        .toThrowError(/Argument 'x' passed to 'argMax' must be numeric tensor/);
  });
});

describe('Reduction: argmin', () => {
  beforeAll(async () => tfwebgpu.ready);

  it('Tensor1D', async () => {
    const a = tf.tensor1d([1, 0, 3, 2]);
    const result = tf.argMin(a);
    tf.test_util.expectArraysEqual(await result.data(), new Float32Array([1]));
  });

  it('one value', async () => {
    const a = tf.tensor1d([10]);
    const result = tf.argMin(a);
    tf.test_util.expectArraysEqual(await result.data(), new Float32Array([0]));
  });

  it('1D large', async () => {
    const n = 250000;
    const values = randomFloats(n);
    const argmin = computeArgMinMax(Array.from(values), 'min');

    const a = tf.tensor1d(values);
    const result = tf.argMin(a);
    expect(result.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(
        await result.data(), new Float32Array([argmin]));
  });

  it('4D large, axis = 3', async () => {
    const n = 250000;
    const values = randomFloats(n);
    const argmin = computeArgMinMax(Array.from(values), 'min');

    const a = tf.tensor4d(values, [1, 1, 1, n]);
    const result = tf.argMin(a, 3);
    expect(result.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(
        await result.data(), new Float32Array([argmin]));
  });

  it('ignores NaNs', async () => {
    const a = tf.tensor1d([5, 0, NaN, -1, 3]);
    const res = tf.argMin(a);
    tf.test_util.expectArraysEqual(await res.data(), new Float32Array([3]));
  });

  it('3D, ignores NaNs', async () => {
    const a = tf.tensor3d([5, 0, NaN, -1, 3], [1, 1, 5]);
    const res = tf.argMin(a, -1);
    tf.test_util.expectArraysEqual(await res.data(), new Float32Array([3]));
  });

  it('2D, no axis specified', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    tf.test_util.expectArraysEqual(
        await tf.argMin(a).data(), new Float32Array([0, 1, 0]));
  });

  it('2D, axis=0', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.argMin(a, 0);

    expect(r.shape).toEqual([3]);
    expect(r.dtype).toBe('int32');
    tf.test_util.expectArraysEqual(await r.data(), new Float32Array([0, 1, 0]));
  });

  it('2D, axis=1', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, -8], [2, 3]);
    const r = tf.argMin(a, 1);
    tf.test_util.expectArraysEqual(await r.data(), new Float32Array([1, 2]));
  });

  it('2D, axis = -1', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, -8], [2, 3]);
    const r = tf.argMin(a, -1);
    tf.test_util.expectArraysEqual(await r.data(), new Float32Array([1, 2]));
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.argMin({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'argMin' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.argMin([1, 0, 3, 2]);
    tf.test_util.expectArraysEqual(await result.data(), new Float32Array([1]));
  });

  it('throws error for string tensor', () => {
    expect(() => tf.argMin(['a']))
        .toThrowError(/Argument 'x' passed to 'argMin' must be numeric tensor/);
  });
});
