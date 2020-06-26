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

describeWithFlags('add', ALL_ENVS, () => {
  it('c + A', async () => {
    const c = tf.scalar(5);
    const a = tf.tensor1d([1, 2, 3]);

    const result = tf.add(c, a);

    expectArraysClose(await result.data(), [6, 7, 8]);
  });

  it('c + A propagates NaNs', async () => {
    const c = tf.scalar(NaN);
    const a = tf.tensor1d([1, 2, 3]);

    const res = tf.add(c, a);

    expectArraysEqual(await res.data(), [NaN, NaN, NaN]);
  });

  it('A + B broadcasting same rank Tensors different shape', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.tensor2d([2, 3], [2, 1]);

    const result = tf.add(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [3, 4, 0, -1];

    expectArraysClose(await result.data(), expected);
  });

  it('A + B broadcast 2D + 1D', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.tensor1d([1, 2]);

    const result = tf.add(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [2, 4, -2, -2];

    expectArraysClose(await result.data(), expected);
  });

  it('A + B', async () => {
    const a = tf.tensor1d([2, 5, 1]);
    const b = tf.tensor1d([4, 2, -1]);

    const result = tf.add(a, b);

    const expected = [6, 7, 0];
    expectArraysClose(await result.data(), expected);
  });

  it('TensorLike', async () => {
    const a = [2, 5, 1];
    const b = [4, 2, -1];

    const result = tf.add(a, b);

    const expected = [6, 7, 0];
    expectArraysClose(await result.data(), expected);
  });

  it('TensorLike chained', async () => {
    const a = tf.tensor1d([2, 5, 1]);
    const b = [4, 2, -1];

    const result = a.add(b);

    const expected = [6, 7, 0];
    expectArraysClose(await result.data(), expected);
  });

  it('A + B propagates NaNs', async () => {
    const a = tf.tensor1d([2, 5, NaN]);
    const b = tf.tensor1d([4, 2, -1]);

    const res = tf.add(a, b);
    expectArraysClose(await res.data(), [6, 7, NaN]);
  });

  it('A + B throws when passed tensors with different shape', () => {
    const a = tf.tensor1d([2, 5, 1, 5]);
    const b = tf.tensor1d([4, 2, -1]);

    expect(() => tf.add(a, b)).toThrowError();
    expect(() => tf.add(b, a)).toThrowError();
  });

  it('2D+scalar broadcast', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.scalar(2);
    const res = tf.add(a, b);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(await res.data(), [3, 4, 5, 6, 7, 8]);
  });

  it('scalar+1D broadcast', async () => {
    const a = tf.scalar(2);
    const b = tf.tensor1d([1, 2, 3, 4, 5, 6]);
    const res = tf.add(a, b);
    expect(res.shape).toEqual([6]);
    expectArraysClose(await res.data(), [3, 4, 5, 6, 7, 8]);
  });

  it('2D+2D broadcast each with 1 dim', async () => {
    const a = tf.tensor2d([1, 2, 5], [1, 3]);
    const b = tf.tensor2d([7, 3], [2, 1]);
    const res = tf.add(a, b);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(await res.data(), [8, 9, 12, 4, 5, 8]);
  });

  it('2D+2D broadcast inner dim of b', async () => {
    const a = tf.tensor2d([1, 2, 5, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([7, 3], [2, 1]);
    const res = tf.add(a, b);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(await res.data(), [8, 9, 12, 7, 8, 9]);
  });

  it('3D+scalar', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1]);
    const b = tf.scalar(-1);
    const res = tf.add(a, b);
    expect(res.shape).toEqual([2, 3, 1]);
    expectArraysClose(await res.data(), [0, 1, 2, 3, 4, 5]);
  });

  it('6D+scalar', async () => {
    const a = tf.range(0, 64).reshape([2, 2, 2, 2, 2, 2]);
    const b = tf.scalar(-1);
    const res = tf.add(a, b);
    expect(res.shape).toEqual([2, 2, 2, 2, 2, 2]);
    const expectedResult = [
      -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
      31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
      47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62
    ];
    expectArraysClose(await res.data(), expectedResult);
  });

  it('6D+2D', async () => {
    const a = tf.range(0, 64).reshape([2, 2, 2, 2, 2, 2]);
    const b = tf.tensor2d([11, 13, 17, 19], [2, 2]);
    const res = tf.add(a, b);
    expect(res.shape).toEqual([2, 2, 2, 2, 2, 2]);
    const expectedResult = [
      11, 14, 19, 22, 15, 18, 23, 26, 19, 22, 27, 30, 23, 26, 31, 34,
      27, 30, 35, 38, 31, 34, 39, 42, 35, 38, 43, 46, 39, 42, 47, 50,
      43, 46, 51, 54, 47, 50, 55, 58, 51, 54, 59, 62, 55, 58, 63, 66,
      59, 62, 67, 70, 63, 66, 71, 74, 67, 70, 75, 78, 71, 74, 79, 82
    ];
    expectArraysClose(await res.data(), expectedResult);
  });

  it('add tensors with 0 in shape', async () => {
    const a = tf.tensor1d([1]);
    const b = tf.tensor3d([], [0, 0, 5]);
    const res = tf.add(a, b);
    expect(res.shape).toEqual([0, 0, 5]);
    expectArraysEqual(await res.data(), []);
  });

  it('gradient: scalar + 1D broadcast', async () => {
    const a = tf.scalar(2);
    const b = tf.tensor1d([3, 4, 5]);
    const dy = tf.tensor1d([7, 8, 9]);

    const grads = tf.grads((a, b) => tf.add(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [7 + 8 + 9]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [7, 8, 9]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(2);
    const b = tf.tensor1d([3, 4, 5]);
    const dy = tf.tensor1d([7, 8, 9]);

    const grads = tf.grads((a, b) => tf.add(a.clone(), b.clone()).clone());
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [7 + 8 + 9]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [7, 8, 9]);
  });

  it('gradient: 2D + 2D broadcast', async () => {
    const a = tf.tensor2d([2, 3], [2, 1]);
    const b = tf.tensor2d([4, 5, 6, 7], [2, 2]);
    const dy = tf.tensor2d([5, 4, 3, 2], [2, 2]);

    const grads = tf.grads((a, b) => tf.add(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [5 + 4, 3 + 2]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [5, 4, 3, 2]);
  });

  it('complex number addition', async () => {
    const real1 = tf.tensor1d([1]);
    const imag1 = tf.tensor1d([2]);
    const complex1 = tf.complex(real1, imag1);

    const real2 = tf.tensor1d([3]);
    const imag2 = tf.tensor1d([4]);
    const complex2 = tf.complex(real2, imag2);

    const result = complex1.add(complex2);

    expect(result.dtype).toBe('complex64');
    expect(result.shape).toEqual([1]);
    expectArraysClose(await result.data(), [4, 6]);
  });

  it('complex number reshape and then addition', async () => {
    const real1 = tf.tensor1d([1]);
    const imag1 = tf.tensor1d([2]);
    const complex1 = tf.complex(real1, imag1);

    const real2 = tf.tensor1d([3]);
    const imag2 = tf.tensor1d([4]);
    const complex2 = tf.complex(real2, imag2);

    const complex1Reshaped = complex1.reshape([1, 1, 1]);
    const complex2Reshaped = complex2.reshape([1, 1, 1]);

    const result = complex1Reshaped.add(complex2Reshaped);

    expect(result.dtype).toBe('complex64');
    expect(result.shape).toEqual([1, 1, 1]);
    expectArraysClose(await result.data(), [4, 6]);
  });

  it('complex number broadcasting addition', async () => {
    const real1 = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const imag1 = tf.tensor2d([10, 20, -30, -40], [2, 2]);
    const complex1 = tf.complex(real1, imag1);

    const real2 = tf.tensor1d([4]);
    const imag2 = tf.tensor1d([5]);
    const complex2 = tf.complex(real2, imag2);

    const result = tf.add(complex1, complex2);

    expect(result.dtype).toEqual('complex64');
    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(
        await result.data(),
        [1 + 4, 10 + 5, 2 + 4, 20 + 5, -3 + 4, -30 + 5, -4 + 4, -40 + 5]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.add({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'a' passed to 'add' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.add(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'add' must be a Tensor/);
  });

  it('upcasts when dtypes dont match', async () => {
    let res = tf.add(tf.scalar(1, 'int32'), tf.scalar(1, 'float32'));
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [2]);

    res = tf.add(tf.scalar(1, 'int32'), tf.scalar(true, 'bool'));
    expect(res.dtype).toBe('int32');
    expectArraysClose(await res.data(), [2]);

    res = tf.add(tf.scalar(1, 'int32'), tf.scalar(false, 'bool'));
    expect(res.dtype).toBe('int32');
    expectArraysClose(await res.data(), [1]);

    res = tf.add(tf.complex(4, 7), tf.scalar(1, 'float32'));
    expect(res.dtype).toBe('complex64');
    expectArraysClose(await res.data(), [5, 7]);

    res = tf.add(tf.complex(4, 7), tf.scalar(1, 'int32'));
    expect(res.dtype).toBe('complex64');
    expectArraysClose(await res.data(), [5, 7]);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.add(5, [1, 2, 3]);
    expectArraysClose(await result.data(), [6, 7, 8]);
  });
});
