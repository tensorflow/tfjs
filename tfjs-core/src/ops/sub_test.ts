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

describeWithFlags('sub', ALL_ENVS, () => {
  it('c - A', async () => {
    const c = tf.scalar(5);
    const a = tf.tensor1d([7, 2, 3]);

    const result = tf.sub(c, a);

    expectArraysClose(await result.data(), [-2, 3, 2]);
  });

  it('A - c', async () => {
    const a = tf.tensor1d([1, 2, -3]);
    const c = tf.scalar(5);

    const result = tf.sub(a, c);

    expectArraysClose(await result.data(), [-4, -3, -8]);
  });

  it('A - c propagates NaNs', async () => {
    const a = tf.tensor1d([1, NaN, 3]);
    const c = tf.scalar(5);

    const res = tf.sub(a, c);

    expectArraysClose(await res.data(), [-4, NaN, -2]);
  });

  it('A - B', async () => {
    const a = tf.tensor1d([2, 5, 1]);
    const b = tf.tensor1d([4, 2, -1]);

    const result = tf.sub(a, b);

    const expected = [-2, 3, 2];
    expectArraysClose(await result.data(), expected);
  });

  it('TensorLike', async () => {
    const a = [2, 5, 1];
    const b = [4, 2, -1];

    const result = tf.sub(a, b);

    const expected = [-2, 3, 2];
    expectArraysClose(await result.data(), expected);
  });

  it('TensorLike chained', async () => {
    const a = tf.tensor1d([2, 5, 1]);
    const b = [4, 2, -1];

    const result = a.sub(b);

    const expected = [-2, 3, 2];
    expectArraysClose(await result.data(), expected);
  });

  it('A - B propagates NaNs', async () => {
    const a = tf.tensor1d([2, 5, 1]);
    const b = tf.tensor1d([4, NaN, -1]);

    const res = tf.sub(a, b);

    expectArraysClose(await res.data(), [-2, NaN, 2]);
  });

  it('A - B throws when passed tensors with different shape', () => {
    const a = tf.tensor1d([2, 5, 1, 5]);
    const b = tf.tensor1d([4, 2, -1]);

    expect(() => tf.sub(a, b)).toThrowError();
    expect(() => tf.sub(b, a)).toThrowError();
  });

  it('A - B broadcasting same rank Tensors different shape', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.tensor2d([2, 3], [2, 1]);

    const result = tf.sub(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [-1, 0, -6, -7];

    expectArraysClose(await result.data(), expected);
  });

  it('A - B broadcast 2D + 1D', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.tensor1d([1, 2]);

    const result = tf.sub(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [0, 0, -4, -6];

    expectArraysClose(await result.data(), expected);
  });

  it('2D-scalar broadcast', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.scalar(2);
    const res = tf.sub(a, b);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(await res.data(), [-1, 0, 1, 2, 3, 4]);
  });

  it('scalar-1D broadcast', async () => {
    const a = tf.scalar(2);
    const b = tf.tensor1d([1, 2, 3, 4, 5, 6]);
    const res = tf.sub(a, b);
    expect(res.shape).toEqual([6]);
    expectArraysClose(await res.data(), [1, 0, -1, -2, -3, -4]);
  });

  it('2D-2D broadcast each with 1 dim', async () => {
    const a = tf.tensor2d([1, 2, 5], [1, 3]);
    const b = tf.tensor2d([7, 3], [2, 1]);
    const res = tf.sub(a, b);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(await res.data(), [-6, -5, -2, -2, -1, 2]);
  });

  it('2D-2D broadcast inner dim of b', async () => {
    const a = tf.tensor2d([1, 2, 5, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([7, 3], [2, 1]);
    const res = tf.sub(a, b);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(await res.data(), [-6, -5, -2, 1, 2, 3]);
  });

  it('3D-scalar', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1]);
    const b = tf.scalar(-1);
    const res = tf.sub(a, b);
    expect(res.shape).toEqual([2, 3, 1]);
    expectArraysClose(await res.data(), [2, 3, 4, 5, 6, 7]);
  });

  it('gradients: basic 1D arrays', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([3, 2, 1]);
    const dy = tf.tensor1d([1, 10, 20]);

    const grads = tf.grads((a, b) => tf.sub(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1, 10, 20]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [-1, -10, -20]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([3, 2, 1]);
    const dy = tf.tensor1d([1, 10, 20]);

    const grads = tf.grads((a, b) => tf.sub(a.clone(), b.clone()).clone());
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1, 10, 20]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [-1, -10, -20]);
  });

  it('gradients: basic 2D arrays', async () => {
    const a = tf.tensor2d([0, 1, 2, 3], [2, 2]);
    const b = tf.tensor2d([3, 2, 1, 0], [2, 2]);
    const dy = tf.tensor2d([1, 10, 15, 20], [2, 2]);

    const grads = tf.grads((a, b) => tf.sub(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1, 10, 15, 20]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [-1, -10, -15, -20]);
  });

  it('gradient: 1D - scalar broadcast', async () => {
    const a = tf.tensor1d([3, 4, 5]);
    const b = tf.scalar(2);
    const dy = tf.tensor1d([7, 8, 9]);

    const grads = tf.grads((a, b) => tf.sub(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [7, 8, 9]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [-7 - 8 - 9]);
  });

  it('gradient: scalar - 1D broadcast', async () => {
    const a = tf.scalar(2);
    const b = tf.tensor1d([3, 4, 5]);
    const dy = tf.tensor1d([7, 8, 9]);

    const grads = tf.grads((a, b) => tf.sub(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [7 + 8 + 9]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [-7, -8, -9]);
  });

  it('gradient: 2D - 2D broadcast', async () => {
    const a = tf.tensor2d([4, 5, 6, 7], [2, 2]);
    const b = tf.tensor2d([2, 3], [2, 1]);
    const dy = tf.tensor2d([5, 4, 3, 2], [2, 2]);

    const grads = tf.grads((a, b) => tf.sub(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [5, 4, 3, 2]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [-5 - 4, -3 - 2]);
  });

  it('complex number subtraction', async () => {
    const real1 = tf.tensor1d([3]);
    const imag1 = tf.tensor1d([5]);
    const complex1 = tf.complex(real1, imag1);

    const real2 = tf.tensor1d([1]);
    const imag2 = tf.tensor1d([0]);
    const complex2 = tf.complex(real2, imag2);

    const result = complex1.sub(complex2);

    expect(result.dtype).toBe('complex64');
    expect(result.shape).toEqual([1]);
    expectArraysClose(await result.data(), [2, 5]);
  });

  it('complex number broadcasting subtraction', async () => {
    const real1 = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const imag1 = tf.tensor2d([10, 20, -30, -40], [2, 2]);
    const complex1 = tf.complex(real1, imag1);

    const real2 = tf.tensor1d([4]);
    const imag2 = tf.tensor1d([5]);
    const complex2 = tf.complex(real2, imag2);

    const result = tf.sub(complex1, complex2);

    expect(result.dtype).toEqual('complex64');
    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(
        await result.data(),
        [1 - 4, 10 - 5, 2 - 4, 20 - 5, -3 - 4, -30 - 5, -4 - 4, -40 - 5]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.sub({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'a' passed to 'sub' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.sub(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'sub' must be a Tensor/);
  });
  it('upcasts when dtypes dont match', async () => {
    let res = tf.sub(tf.scalar(1, 'int32'), tf.scalar(1, 'float32'));
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [0]);

    res = tf.sub(tf.scalar(1, 'int32'), tf.scalar(true, 'bool'));
    expect(res.dtype).toBe('int32');
    expectArraysClose(await res.data(), [0]);

    res = tf.sub(tf.scalar(1, 'int32'), tf.scalar(false, 'bool'));
    expect(res.dtype).toBe('int32');
    expectArraysClose(await res.data(), [1]);

    res = tf.sub(tf.complex(4, 7), tf.scalar(1, 'float32'));
    expect(res.dtype).toBe('complex64');
    expectArraysClose(await res.data(), [3, 7]);

    res = tf.sub(tf.complex(4, 7), tf.scalar(1, 'int32'));
    expect(res.dtype).toBe('complex64');
    expectArraysClose(await res.data(), [3, 7]);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.sub(5, [7, 2, 3]);
    expectArraysClose(await result.data(), [-2, 3, 2]);
  });
});
