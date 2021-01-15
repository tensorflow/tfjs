/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

describeWithFlags('div', ALL_ENVS, () => {
  it('same shape', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const c = tf.tensor2d([1, 2, 3, 4, 2, 5], [2, 3]);

    const r = tf.div(a, c);

    expectArraysClose(await r.data(), [1, 1, 1, 1, 2.5, 6 / 5]);
  });

  it('TensorLike', async () => {
    const a = [0, 1, -2, -4, 4, -4];
    const b = [0.15, 0.2, 0.25, 0.5, 0.7, 1.2];
    const result = tf.div(a, b);

    expect(result.shape).toEqual([6]);
    expectArraysClose(
        await result.data(),
        [0, 5.0, -8.0, -8.0, 5.714285850524902, -3.3333332538604736]);
  });

  it('TensorLike chained', async () => {
    const a = tf.tensor1d([0, 1, -2, -4, 4, -4]);
    const b = [0.15, 0.2, 0.25, 0.5, 0.7, 1.2];
    const result = a.div(b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(
        await result.data(),
        [0, 5.0, -8.0, -8.0, 5.714285850524902, -3.3333332538604736]);
  });

  it('division by zero results in infinity', async () => {
    const r = tf.div(1, 0);
    const rData = await r.data();

    expect(Array.from(rData)).toEqual([Infinity]);
  });

  it('integer division implements floor divide', async () => {
    const a = tf.tensor1d([-6, -6, -5, -4, -3, -3, 3, 3, 2], 'int32');
    const c = tf.tensor1d([-2, 2, 3, 2, -3, 3, 2, 3, 2], 'int32');

    const r = tf.div(a, c);

    expect(r.dtype).toEqual('int32');
    expectArraysClose(await r.data(), [3, -3, -2, -2, 1, -1, 1, 1, 1]);
  });

  it('integer division broadcasts', async () => {
    const a = tf.tensor1d([-5, -4, 3, 2], 'int32');
    const c = tf.scalar(2, 'int32');

    const r = tf.div(a, c);

    expect(r.dtype).toEqual('int32');
    expectArraysClose(await r.data(), [-3, -2, 1, 1]);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor2d([1, 2], [2, 1]);
    const c = tf.tensor2d([3, NaN], [2, 1]);

    const r = tf.div(a, c);

    expectArraysClose(await r.data(), [1 / 3, NaN]);
  });

  it('broadcasting same rank Tensors different shape', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.tensor2d([2, 3], [2, 1]);

    const result = tf.div(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [1 / 2, 1, -1, -4 / 3];

    expectArraysClose(await result.data(), expected);
  });

  it('broadcast scalar', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = [2];

    const result = tf.div(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [0.5, 1, 1.5, 2];

    expectArraysClose(await result.data(), expected);
  });

  it('broadcast 2D + 1D', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.tensor1d([1, 2]);

    const result = tf.div(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [1, 1, -3, -2];

    expectArraysClose(await result.data(), expected);
  });

  it('upcasts when dtypes dont match', async () => {
    let res = tf.div(tf.scalar(6, 'int32'), tf.scalar(3, 'float32'));
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [2]);

    res = tf.div(tf.scalar(6, 'int32'), tf.scalar(true, 'bool'));
    expect(res.dtype).toBe('int32');
    expectArraysClose(await res.data(), [6]);
  });

  it('throws when passed tensors of different shapes', () => {
    const a = tf.tensor2d([1, 2, -3, -4, 5, 6], [2, 3]);
    const b = tf.tensor2d([5, 3, 4, -7], [2, 2]);

    expect(() => tf.div(a, b)).toThrowError();
    expect(() => tf.div(b, a)).toThrowError();
  });

  it('scalar divided by array', async () => {
    const c = tf.scalar(2);
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);

    const r = tf.div(c, a);

    expectArraysClose(
        await r.data(), [2 / 1, 2 / 2, 2 / 3, 2 / 4, 2 / 5, 2 / 6]);
  });

  it('scalar divided by array propagates NaNs', async () => {
    const c = tf.scalar(NaN);
    const a = tf.tensor2d([1, 2, 3], [1, 3]);

    const r = tf.div(c, a);

    expectArraysEqual(await r.data(), [NaN, NaN, NaN]);
  });

  it('array divided by scalar', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const c = tf.scalar(2);

    const r = tf.div(a, c);

    expectArraysClose(
        await r.data(), [1 / 2, 2 / 2, 3 / 2, 4 / 2, 5 / 2, 6 / 2]);
  });

  it('array divided by scalar propagates NaNs', async () => {
    const a = tf.tensor2d([1, 2, NaN], [1, 3]);
    const c = tf.scalar(2);

    const r = tf.div(a, c);
    expectArraysClose(await r.data(), [1 / 2, 2 / 2, NaN]);
  });

  it('gradient: Scalar', async () => {
    const a = tf.scalar(5);
    const b = tf.scalar(2);
    const dy = tf.scalar(4);

    const before = tf.memory().numTensors;
    const grads = tf.grads((a, b) => tf.div(a, b));
    const [da, db] = grads([a, b], dy);
    const now = tf.memory().numTensors;
    expect(now).toBe(before + 2);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [4 / 2]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [-4 * 5 / (2 * 2)]);
  });

  it('gradient with clones', async () => {
    const grads = tf.grads((a, b) => tf.div(a.clone(), b.clone()).clone());
    const [da, db] = grads([5, 2]);
    expect(da.shape).toEqual([]);
    expect(db.shape).toEqual([]);
    expectArraysClose(await da.data(), [1 / 2]);
    expectArraysClose(await db.data(), [-5 / 4]);
  });

  it('gradient: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([3, 4, 5]);
    const dy = tf.tensor1d([1, 10, 20]);

    const grads = tf.grads((a, b) => tf.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1 / 3, 10 / 4, 20 / 5]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        await db.data(), [-1 * 1 / 9, -10 * 2 / 16, -20 * 3 / 25]);
  });

  it('gradient: Tensor1D with int32', async () => {
    const a = tf.tensor1d([1, 2, 3], 'int32');
    const b = tf.tensor1d([3, 4, 5], 'int32');
    const dy = tf.tensor1d([1, 10, 20]);

    const grads = tf.grads((a, b) => tf.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1 / 3, 10 / 4, 20 / 5]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        await db.data(), [-1 * 1 / 9, -10 * 2 / 16, -20 * 3 / 25]);
  });

  it('gradient: 1d<int32> with 1d<bool> ', async () => {
    const a = tf.tensor1d([true, false, true], 'bool');
    const b = tf.tensor1d([1, 2, 3], 'int32');
    const dy = tf.tensor1d([1, 19, 20]);

    const grads = tf.grads((a, b) => tf.div(a.toInt(), b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1, 19 / 2, 20 / 3]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [-1 / 1, 0, -20 / 9]);
  });

  it('gradient: Tensor2D', async () => {
    const a = tf.tensor2d([3, 1, 2, 3], [2, 2]);
    const b = tf.tensor2d([1, 3, 4, 5], [2, 2]);
    const dy = tf.tensor2d([1, 10, 15, 20], [2, 2]);

    const grads = tf.grads((a, b) => tf.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1 / 1, 10 / 3, 15 / 4, 20 / 5]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        await db.data(), [-1 * 3 / 1, -10 * 1 / 9, -15 * 2 / 16, -20 * 3 / 25]);
  });

  it('gradient: scalar / Tensor1D', async () => {
    const a = tf.scalar(2);
    const b = tf.tensor1d([3, 4, 5]);
    const dy = tf.tensor1d([6, 7, 8]);

    const grads = tf.grads((a, b) => tf.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [6 / 3 + 7 / 4 + 8 / 5]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [-6 * 2 / 9, -7 * 2 / 16, -8 * 2 / 25]);
  });

  it('gradient: Tensor2D / scalar', async () => {
    const a = tf.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const b = tf.scalar(2);
    const dy = tf.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = tf.grads((a, b) => tf.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [6 / 2, 7 / 2, 8 / 2, 9 / 2]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        await db.data(), [-6 * 2 / 4 + -7 * 3 / 4 + -8 * 4 / 4 + -9 * 5 / 4]);
  });

  it('gradient: Tensor2D / Tensor2D w/ broadcast', async () => {
    const a = tf.tensor2d([3, 4], [2, 1]);
    const b = tf.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const dy = tf.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = tf.grads((a, b) => tf.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [6 / 2 + 7 / 3, 8 / 4 + 9 / 5]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        await db.data(), [-6 * 3 / 4, -7 * 3 / 9, -8 * 4 / 16, -9 * 4 / 25]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.div({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'a' passed to 'div' must be a Tensor/);
  });

  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.div(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'div' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.div([[1, 2, 3], [4, 5, 6]], 2);
    expect(r.shape).toEqual([2, 3]);
    expectArraysClose(
        await r.data(), [1 / 2, 2 / 2, 3 / 2, 4 / 2, 5 / 2, 6 / 2]);
  });
});

describeWithFlags('mul', ALL_ENVS, () => {
  it('same-shaped tensors', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.tensor2d([5, 3, 4, -7], [2, 2]);
    const expected = [5, 6, -12, 28];
    const result = tf.mul(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(await result.data(), expected);
  });

  it('TensorLike', async () => {
    const a = [[1, 2], [-3, -4]];
    const b = [[5, 3], [4, -7]];
    const expected = [5, 6, -12, 28];
    const result = tf.mul(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(await result.data(), expected);
  });

  it('TensorLike chained', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = [[5, 3], [4, -7]];
    const expected = [5, 6, -12, 28];
    const result = a.mul(b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(await result.data(), expected);
  });

  it('broadcasting tensors', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.scalar(2);
    const expected = [2, 4, -6, -8];
    const result = tf.mul(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(await result.data(), expected);
  });

  it('broadcasting same rank Tensors different shape', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.tensor2d([2, 3], [2, 1]);

    const result = tf.mul(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [2, 4, -9, -12];

    expectArraysClose(await result.data(), expected);
  });

  it('broadcast 2D + 1D', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.tensor1d([1, 2]);

    const result = tf.mul(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [1, 4, -3, -8];

    expectArraysClose(await result.data(), expected);
  });

  it('broadcast 5D + 2D', async () => {
    const a = tf.range(1, 33).reshape([2, 2, 2, 2, 2]);
    const b = tf.tensor([2, 3], [2, 1]);
    const result = tf.mul(a, b);
    expect(result.shape).toEqual([2, 2, 2, 2, 2]);
    const expected = [
      2,  4,  9,  12, 10, 12, 21, 24, 18, 20, 33, 36, 26, 28, 45, 48,
      34, 36, 57, 60, 42, 44, 69, 72, 50, 52, 81, 84, 58, 60, 93, 96
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('broadcast 6D + 2D', async () => {
    const a = tf.range(1, 65).reshape([2, 2, 2, 2, 2, 2]);
    const b = tf.tensor([2, 3], [2, 1]);
    const result = tf.mul(a, b);
    expect(result.shape).toEqual([2, 2, 2, 2, 2, 2]);
    const expected = [
      2,   4,   9,   12,  10,  12,  21,  24,  18,  20,  33,  36,  26,
      28,  45,  48,  34,  36,  57,  60,  42,  44,  69,  72,  50,  52,
      81,  84,  58,  60,  93,  96,  66,  68,  105, 108, 74,  76,  117,
      120, 82,  84,  129, 132, 90,  92,  141, 144, 98,  100, 153, 156,
      106, 108, 165, 168, 114, 116, 177, 180, 122, 124, 189, 192
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('gradient: Scalar', async () => {
    const a = tf.scalar(5);
    const b = tf.scalar(2);
    const dy = tf.scalar(4);

    const grads = tf.grads((a, b) => tf.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), await b.mul(dy).data());

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), await a.mul(dy).data());
  });

  it('gradient with clones', async () => {
    const grads = tf.grads((a, b) => tf.mul(a.clone(), b.clone()).clone());
    const [da, db] = grads([4, 2]);

    expect(da.shape).toEqual([]);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), 2);

    expect(db.shape).toEqual([]);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), 4);
  });

  it('gradient: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([3, 4, 5]);
    const dy = tf.tensor1d([1, 10, 20]);

    const grads = tf.grads((a, b) => tf.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [3 * 1, 4 * 10, 5 * 20]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [1 * 1, 2 * 10, 3 * 20]);
  });

  it('gradient: Tensor1D with dtype int32', async () => {
    const a = tf.tensor1d([1, 2, 3], 'int32');
    const b = tf.tensor1d([3, 4, 5], 'int32');
    const dy = tf.tensor1d([1, 10, 20]);

    const grads = tf.grads((a, b) => tf.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [3 * 1, 4 * 10, 5 * 20]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [1 * 1, 2 * 10, 3 * 20]);
  });

  it('gradient: Tensor2D', async () => {
    const a = tf.tensor2d([3, 1, 2, 3], [2, 2]);
    const b = tf.tensor2d([1, 3, 4, 5], [2, 2]);
    const dy = tf.tensor2d([1, 10, 15, 20], [2, 2]);

    const grads = tf.grads((a, b) => tf.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1 * 1, 3 * 10, 4 * 15, 5 * 20]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [3 * 1, 1 * 10, 2 * 15, 3 * 20]);
  });

  it('gradient: scalar * Tensor1D', async () => {
    const a = tf.scalar(2);
    const b = tf.tensor1d([3, 4, 5]);
    const dy = tf.tensor1d([6, 7, 8]);

    const grads = tf.grads((a, b) => tf.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [3 * 6 + 4 * 7 + 5 * 8]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [2 * 6, 2 * 7, 2 * 8]);
  });

  it('gradient: Tensor2D * scalar', async () => {
    const a = tf.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const b = tf.scalar(2);
    const dy = tf.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = tf.grads((a, b) => tf.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [2 * 6, 2 * 7, 2 * 8, 2 * 9]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [2 * 6 + 3 * 7 + 4 * 8 + 5 * 9]);
  });

  it('gradient: Tensor2D * Tensor2D w/ broadcast', async () => {
    const a = tf.tensor2d([3, 4], [2, 1]);
    const b = tf.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const dy = tf.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = tf.grads((a, b) => tf.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [2 * 6 + 3 * 7, 4 * 8 + 5 * 9]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [6 * 3, 7 * 3, 8 * 4, 9 * 4]);
  });

  it('complex number multiplication', async () => {
    const real1 = tf.tensor1d([2]);
    const imag1 = tf.tensor1d([3]);
    const complex1 = tf.complex(real1, imag1);

    const real2 = tf.tensor1d([4]);
    const imag2 = tf.tensor1d([5]);
    const complex2 = tf.complex(real2, imag2);

    const result = complex1.mul(complex2);

    expect(result.dtype).toBe('complex64');
    expect(result.shape).toEqual([1]);
    expectArraysClose(await result.data(), [2 * 4 - 3 * 5, 2 * 5 + 3 * 4]);
  });

  it('complex number broadcasting multiplication', async () => {
    const real1 = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const imag1 = tf.tensor2d([10, 20, -30, -40], [2, 2]);
    const complex1 = tf.complex(real1, imag1);

    const real2 = tf.tensor1d([4]);
    const imag2 = tf.tensor1d([5]);
    const complex2 = tf.complex(real2, imag2);

    const result = tf.mul(complex1, complex2);

    expect(result.dtype).toEqual('complex64');
    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(await result.data(), [
      1 * 4 - 10 * 5, 1 * 5 + 10 * 4, 2 * 4 - 20 * 5, 2 * 5 + 20 * 4,
      -3 * 4 + 30 * 5, -3 * 5 + -30 * 4, -4 * 4 + 40 * 5, -4 * 5 + -40 * 4
    ]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.mul({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'a' passed to 'mul' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.mul(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'mul' must be a Tensor/);
  });
  it('upcasts when dtypes dont match', async () => {
    let res = tf.mul(tf.scalar(2, 'int32'), tf.scalar(3, 'float32'));
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [6]);

    res = tf.mul(tf.scalar(2, 'int32'), tf.scalar(true, 'bool'));
    expect(res.dtype).toBe('int32');
    expectArraysClose(await res.data(), [2]);

    res = tf.mul(tf.scalar(2, 'int32'), tf.scalar(false, 'bool'));
    expect(res.dtype).toBe('int32');
    expectArraysClose(await res.data(), [0]);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.mul([[1, 2], [-3, -4]], 2);
    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(await result.data(), [2, 4, -6, -8]);
  });
});

describeWithFlags('pow', ALL_ENVS, () => {
  it('same-shaped tensors', async () => {
    const a = tf.tensor2d([1, -2, -3, 0, 7, 1], [2, 3]);
    const b = tf.tensor2d([5, 3, 4, 5, 2, -3], [2, 3], 'int32');
    const expected = [1, -8, 81, 0, 49, 1];
    const result = tf.pow(a, b);

    expect(result.shape).toEqual([2, 3]);
    expectArraysClose(await result.data(), expected, 0.01);
  });

  it('TensorLike', async () => {
    const a = [1, 2, 3];
    const exp = 2;

    const result = tf.pow(a, exp);

    expect(result.shape).toEqual([3]);
    expect(result.dtype).toBe('float32');
    expectArraysClose(await result.data(), [1, 4, 9]);
  });

  it('TensorLike chained', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const exp = 2;

    const result = a.pow(exp);

    expect(result.shape).toEqual([3]);
    expect(result.dtype).toBe('float32');
    expectArraysClose(await result.data(), [1, 4, 9]);
  });

  it('int32^int32 returns int32', async () => {
    const a = tf.tensor1d([1, 2, 3], 'int32');
    const exp = tf.scalar(2, 'int32');

    const result = tf.pow(a, exp);

    expect(result.shape).toEqual([3]);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), [1, 4, 9]);
  });

  it('different-shaped tensors', async () => {
    const a = tf.tensor2d([1, -2, -3, 0, 7, 1], [2, 3]);
    const b = tf.scalar(2, 'int32');
    const expected = [1, 4, 9, 0, 49, 1];
    const result = tf.pow(a, b);

    expect(result.shape).toEqual([2, 3]);
    expectArraysClose(await result.data(), expected, 0.05);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor2d([NaN, 3, NaN, 0], [2, 2]);
    const b = tf.tensor2d([1, 3, 2, 3], [2, 2], 'int32');

    const result = tf.pow(a, b);
    expectArraysClose(await result.data(), [NaN, 27, NaN, 0], 0.05);
  });

  it('exponent of 0 returns 1', async () => {
    const a = tf.tensor1d([-2, -1, 0, 1, 2]);
    const b = tf.scalar(0);

    const result = tf.pow(a, b);
    expectArraysClose(await result.data(), [1, 1, 1, 1, 1]);
  });

  it('handles non int32 exponent param', async () => {
    const a = tf.tensor1d([2, 4]);
    const b = tf.tensor1d([.5, 1.2]);

    const result = tf.pow(a, b);
    const expected = [Math.pow(2, 0.5), Math.pow(4, 1.2)];
    expectArraysClose(await result.data(), expected);
  });

  it('broadcasting same rank Tensors different shape', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.tensor2d([2, 1], [2, 1], 'int32');

    const result = tf.pow(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [1, 4, -3, -4];

    expectArraysClose(await result.data(), expected);
  });

  it('broadcast 2D + 1D', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.tensor1d([1, 2], 'int32');

    const result = tf.pow(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [1, 4, -3, 16];

    expectArraysClose(await result.data(), expected);
  });

  it('gradients: Scalar ^ Scalar', async () => {
    const a = tf.scalar(5);
    const b = tf.scalar(2, 'int32');
    const dy = tf.scalar(3);

    const grads = tf.grads((a, b) => tf.pow(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [2 * 5 * 3]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [3 * Math.pow(5, 2) * Math.log(5)]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(5);
    const b = tf.scalar(2, 'int32');

    const grads = tf.grads((a, b) => tf.pow(a.clone(), b.clone()).clone());
    const [da, db] = grads([a, b]);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [2 * 5]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [Math.pow(5, 2) * Math.log(5)]);
  });

  it('gradients: x ^ 2 where x = 0', async () => {
    const f = (x: tf.Scalar) => x.pow(tf.scalar(2)).asScalar();
    const g = tf.grad(f)(tf.scalar(0));

    expectArraysClose(await g.data(), [0]);
  });

  it('gradients: Scalar ^ Scalar fractional exponent', async () => {
    const a = tf.scalar(4.0);
    const b = tf.scalar(1.5);
    const dy = tf.scalar(3.0);

    const grads = tf.grads((a, b) => tf.pow(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1.5 * Math.pow(4, 0.5) * 3]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        await db.data(), [3.0 * Math.pow(4, 1.5) * Math.log(4.0)]);
  });

  it('gradients: Tensor ^ Tensor', async () => {
    const a = tf.tensor1d([-1, .5, 2]);
    const b = tf.tensor1d([3, 2, -1], 'int32');
    const dy = tf.tensor1d([1, 5, 10]);

    const grads = tf.grads((a, b) => tf.pow(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(
        await da.data(),
        [
          3 * Math.pow(-1, 2) * 1, 2 * Math.pow(.5, 1) * 5,
          -1 * Math.pow(2, -2) * 10
        ],
        1e-1);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [
      0, 5 * Math.pow(.5, 2) * Math.log(.5), 10 * Math.pow(2, -1) * Math.log(2)
    ]);
  });

  it('gradient wrt exponent with negative base', async () => {
    const a = tf.tensor1d([-1, -.5, -2.7]);
    const b = tf.tensor1d([3, 2, -1], 'int32');
    const dy = tf.tensor1d([1, 1, 1]);

    const grads = tf.grads((a, b) => tf.pow(a, b));
    const [, db] = grads([a, b], dy);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [0, 0, 0]);
  });

  it('gradient: scalar / Tensor1D', async () => {
    const a = tf.scalar(2);
    const b = tf.tensor1d([3, 4, 5]);
    const dy = tf.tensor1d([6, 7, 8]);

    const grads = tf.grads((a, b) => tf.pow(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [
      6 * 3 * Math.pow(2, 2) + 7 * 4 * Math.pow(2, 3) + 8 * 5 * Math.pow(2, 4)
    ]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [
      6 * Math.pow(2, 3) * Math.log(2), 7 * Math.pow(2, 4) * Math.log(2),
      8 * Math.pow(2, 5) * Math.log(2)
    ]);
  });

  it('gradient: Tensor2D / scalar', async () => {
    const a = tf.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const b = tf.scalar(2);
    const dy = tf.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = tf.grads((a, b) => tf.pow(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [
      6 * 2 * Math.pow(2, 1), 7 * 2 * Math.pow(3, 1), 8 * 2 * Math.pow(4, 1),
      9 * 2 * Math.pow(5, 1)
    ]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        await db.data(),
        [6 * Math.pow(2, 2) * Math.log(2) + 7 * Math.pow(3, 2) * Math.log(3) +
         8 * Math.pow(4, 2) * Math.log(4) + 9 * Math.pow(5, 2) * Math.log(5)]);
  });

  it('gradient: Tensor2D / Tensor2D w/ broadcast', async () => {
    const a = tf.tensor2d([3, 4], [2, 1]);
    const b = tf.tensor2d([[2, 3], [.4, .5]], [2, 2]);
    const dy = tf.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = tf.grads((a, b) => tf.pow(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [
      6 * 2 * Math.pow(3, 1) + 7 * 3 * Math.pow(3, 2),
      8 * .4 * Math.pow(4, .4 - 1) + 9 * .5 * Math.pow(4, .5 - 1)
    ]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [
      6 * Math.pow(3, 2) * Math.log(3), 7 * Math.pow(3, 3) * Math.log(3),
      8 * Math.pow(4, .4) * Math.log(4), 9 * Math.pow(4, .5) * Math.log(4)
    ]);
  });

  it('throws when passed base as a non-tensor', () => {
    expect(() => tf.pow({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'base' passed to 'pow' must be a Tensor/);
  });
  it('throws when passed exp as a non-tensor', () => {
    expect(() => tf.pow(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'exp' passed to 'pow' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.pow([1, 2, 3], 2);

    expect(result.shape).toEqual([3]);
    expect(result.dtype).toBe('float32');
    expectArraysClose(await result.data(), [1, 4, 9]);
  });

  it('negative base and whole exponent not NaN', async () => {
    const a = tf.tensor1d([-2, -3, -4], 'float32');
    const b = tf.tensor1d([2, -3, 4], 'float32');

    const expected = [Math.pow(-2, 2), Math.pow(-3, -3), Math.pow(-4, 4)];
    const result = tf.pow(a, b);

    expectArraysClose(await result.data(), expected);
  });

  it('negative base and fract exponent NaN', async () => {
    const a = tf.tensor1d([-2, -3, -4], 'float32');
    const b = tf.tensor1d([2.1, -3.01, 4.1], 'float32');

    const expected = [NaN, NaN, NaN];
    const result = tf.pow(a, b);

    expectArraysClose(await result.data(), expected);
  });
});
