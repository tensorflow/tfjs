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

import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
import {MATMUL_SHARED_DIM_THRESHOLD} from '../kernels/backend_webgl';
import {ALL_ENVS, expectArraysClose, expectNumbersClose, WEBGL_ENVS} from '../test_util';
import {Rank} from '../types';

describeWithFlags('packed matmul', WEBGL_ENVS, () => {
  it('should not leak memory', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
    const b = tf.tensor2d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [3, 5]);

    const startNumBytes = tf.memory().numBytes;
    tf.matMul(a, b);
    const endNumBytes = tf.memory().numBytes;

    expect(endNumBytes - startNumBytes).toEqual(60);
  });

  it('should work when input matrix dimensions are not divisible by 2', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
    const b = tf.tensor2d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [3, 5]);

    const c = tf.matMul(a, b);

    expect(c.shape).toEqual([3, 5]);
    expectArraysClose(
        c,
        [46, 52, 58, 64, 70, 100, 115, 130, 145, 160, 154, 178, 202, 226, 250]);
  });

  it('should work when output texture shape != physical shape', () => {
    const sharedDim = 16000;
    const a = tf.buffer<Rank.R2>([2, sharedDim], 'float32');
    const b = tf.buffer<Rank.R2>([sharedDim, 2], 'float32');

    a.set(1, 0, sharedDim - 1);
    a.set(1, 0, sharedDim - 2);
    a.set(1, 1, sharedDim - 1);
    b.set(1, sharedDim - 1, 0);
    b.set(1, sharedDim - 2, 0);

    const c = tf.matMul(a.toTensor(), b.toTensor());
    const expected = [2, 0, 1, 0];
    expectArraysClose(c, expected);
  });

  it('should work when input texture shapes != physical shape', () => {
    const maxTextureSize = tf.ENV.get('WEBGL_MAX_TEXTURE_SIZE');
    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', 5);
    const a = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 12]);
    const b = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [12, 1]);

    const c = tf.matMul(a, b);

    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', maxTextureSize);

    expectArraysClose(c, [572]);
  });

  it('A x B', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.matMul(a, b);

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(c, [0, 8, -3, 20]);
  });

  it('A x B^t', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([1, 0, 2, 4, 3, 0], [2, 3]);

    const transposeA = false;
    const transposeB = true;
    const c = tf.matMul(a, b, transposeA, transposeB);

    const expected = [7, 10, 16, 31];
    expectArraysClose(c, expected);
  });

  it('A^t x B', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([1, 0, 2, 4, 3, 0], [2, 3]);

    const transposeA = true;
    const transposeB = false;
    const c = tf.matMul(a, b, transposeA, transposeB);

    const expected = [17, 12, 2, 22, 15, 4, 27, 18, 6];
    expectArraysClose(c, expected);
  });

  it('A^t x B^t', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    const b = tf.tensor2d([1, 0, 2, 4, 3, 0], [2, 3]);

    const transposeA = true;
    const transposeB = true;
    const c = tf.matMul(a, b, transposeA, transposeB);

    const expected = [11, 13, 14, 20];
    expectArraysClose(c, expected);
  });

  it('works when followed by an op that requires unpacked inputs', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.matMul(a, b);
    const d = tf.add(c, 1);

    expectArraysClose(d, [1, 9, -2, 21]);
  });

  // tslint:disable-next-line:max-line-length
  it('works when followed by a reshape that changes texture layout, and then an unpacked op',
     () => {
       const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 1]);
       const b = tf.tensor2d([1], [1, 1]);
       const c = tf.matMul(a, b);

       const d = tf.reshape(c, [1, 3, 3, 1]);
       const e = tf.add(d, 1);
       expectArraysClose(e, [2, 3, 4, 5, 6, 7, 8, 9, 10]);
     });

  it('works when preceded by an op that requires packed inputs', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.add(a, 1);
    const d = tf.matMul(b, c);

    expectArraysClose(d, [5, 6, 7, 4, 3, 2, 9, 12, 15]);
  });
});

describeWithFlags('matmul', ALL_ENVS, () => {
  it('A x B', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.matMul(a, b);

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(c, [0, 8, -3, 20]);
  });

  it('A x B^t', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([1, 0, 2, 4, 3, 0], [2, 3]);

    const transposeA = false;
    const transposeB = true;
    const c = tf.matMul(a, b, transposeA, transposeB);

    const expected = [7, 10, 16, 31];
    expectArraysClose(c, expected);
  });

  it('A^t x B', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([1, 0, 2, 4, 3, 0], [2, 3]);

    const transposeA = true;
    const transposeB = false;
    const c = tf.matMul(a, b, transposeA, transposeB);

    const expected = [17, 12, 2, 22, 15, 4, 27, 18, 6];
    expectArraysClose(c, expected);
  });

  it('A^t x B^t', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    const b = tf.tensor2d([1, 0, 2, 4, 3, 0], [2, 3]);

    const transposeA = true;
    const transposeB = true;
    const c = tf.matMul(a, b, transposeA, transposeB);

    const expected = [11, 13, 14, 20];
    expectArraysClose(c, expected);
  });

  it('A x B^t shapes do not match', () => {
    const a = tf.zeros<Rank.R2>([2, 3]);
    const b = tf.zeros<Rank.R2>([3, 2]);

    const f = () => {
      const transposeA = false;
      const transposeB = true;
      tf.matMul(a, b, transposeA, transposeB);
    };
    expect(f).toThrowError();
  });

  it('A^t x B shapes do not match', () => {
    const a = tf.zeros<Rank.R2>([2, 3]);
    const b = tf.zeros<Rank.R2>([3, 2]);

    const f = () => {
      const transposeA = true;
      const transposeB = false;
      tf.matMul(a, b, transposeA, transposeB);
    };
    expect(f).toThrowError();
  });

  it('A^t x B^t shapes do not match', () => {
    const a = tf.zeros<Rank.R2>([3, 2]);
    const b = tf.zeros<Rank.R2>([3, 2]);

    const f = () => {
      const transposeA = true;
      const transposeB = true;
      tf.matMul(a, b, transposeA, transposeB);
    };
    expect(f).toThrowError();
  });

  it('matmul throws when inner dimensions dont match', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1, 2, 2], [4, 2]);

    expect(() => tf.matMul(a, b)).toThrowError();
  });

  it('matmul throws when passed non matrices', () => {
    // tslint:disable-next-line:no-any
    const a: any =
        tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1, 2, 2], [4, 2]);

    expect(() => tf.matMul(a, b)).toThrowError();
    expect(() => tf.matMul(b, a)).toThrowError();
  });

  it('matmul throws when passed a vector', () => {
    // tslint:disable-next-line:no-any
    const v: any = tf.tensor1d([2, 3]);
    const matrix = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    expect(() => tf.matMul(matrix, v)).toThrowError();
  });

  it('Vector times matrix', () => {
    const v = tf.tensor1d([2, 3]);
    const matrix = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const result = tf.dot(v, matrix);

    const expected = [11, 16];
    expectArraysClose(result, expected);
  });

  it('Vector times matrix with implicit reshape', () => {
    const v = tf.tensor1d([2, 3]);

    const matrix = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const result = tf.dot(v, matrix);

    const expected = [11, 16];
    expectArraysClose(result, expected);
  });

  it('Matrix times vector', () => {
    const matrix = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const v = tf.tensor1d([2, 3]);
    const result = tf.dot(matrix, v);

    const expected = [8, 18];
    expectArraysClose(result, expected);
  });

  it('batched matmul with the matrices being vectors', () => {
    const batch = 3;
    const sharedDim = MATMUL_SHARED_DIM_THRESHOLD + 1;
    const values = new Float32Array(batch * sharedDim);
    values[10] = 2;

    const a = tf.tensor(values, [batch, 1, sharedDim]);
    const b = tf.tensor(values, [batch, sharedDim, 1]);
    const result = tf.matMul(a, b);
    expect(result.shape).toEqual([batch, 1, 1]);
    expectArraysClose(result, [4, 0, 0]);
  });

  it('batched matmul with the matrices being vectors transposedA', () => {
    const batch = 3;
    const sharedDim = MATMUL_SHARED_DIM_THRESHOLD + 1;
    const values = new Float32Array(batch * sharedDim);
    values[10] = 2;

    const a = tf.tensor(values, [batch, sharedDim, 1]);
    const b = tf.tensor(values, [batch, sharedDim, 1]);
    const transposeA = true;
    const transposeB = false;
    const result = tf.matMul(a, b, transposeA, transposeB);
    expect(result.shape).toEqual([batch, 1, 1]);
    expectArraysClose(result, [4, 0, 0]);
  });

  it('batched matmul with the matrices being vectors transposedB', () => {
    const batch = 3;
    const sharedDim = MATMUL_SHARED_DIM_THRESHOLD + 1;
    const values = new Float32Array(batch * sharedDim);
    values[10] = 2;

    const a = tf.tensor(values, [batch, 1, sharedDim]);
    const b = tf.tensor(values, [batch, 1, sharedDim]);
    const transposeA = false;
    const transposeB = true;
    const result = tf.matMul(a, b, transposeA, transposeB);
    expect(result.shape).toEqual([batch, 1, 1]);
    expectArraysClose(result, [4, 0, 0]);
  });

  it('batched matmul with matrix x vector', () => {
    const batch = 3;
    const sharedDim = MATMUL_SHARED_DIM_THRESHOLD + 1;
    const values = new Float32Array(batch * sharedDim);
    values[10] = 2;

    const a = tf.ones([batch, 2, sharedDim]);
    const b = tf.tensor(values, [batch, sharedDim, 1]);
    const result = tf.matMul(a, b);
    expect(result.shape).toEqual([batch, 2, 1]);
    expectArraysClose(result, [2, 2, 0, 0, 0, 0]);
  });

  it('batched matmul with matrix x vector transposedA', () => {
    const batch = 3;
    const sharedDim = MATMUL_SHARED_DIM_THRESHOLD + 1;
    const values = new Float32Array(batch * sharedDim);
    values[10] = 2;

    const a = tf.ones([batch, sharedDim, 2]);
    const b = tf.tensor(values, [batch, sharedDim, 1]);
    const transposeA = true;
    const transposeB = false;
    const result = tf.matMul(a, b, transposeA, transposeB);
    expect(result.shape).toEqual([batch, 2, 1]);
    expectArraysClose(result, [2, 2, 0, 0, 0, 0]);
  });

  it('batched matmul with matrix x vector transposedB', () => {
    const batch = 3;
    const sharedDim = MATMUL_SHARED_DIM_THRESHOLD + 1;
    const values = new Float32Array(batch * sharedDim);
    values[10] = 2;

    const a = tf.ones([batch, 2, sharedDim]);
    const b = tf.tensor(values, [batch, 1, sharedDim]);
    const transposeA = false;
    const transposeB = true;
    const result = tf.matMul(a, b, transposeA, transposeB);
    expect(result.shape).toEqual([batch, 2, 1]);
    expectArraysClose(result, [2, 2, 0, 0, 0, 0]);
  });

  it('batched matmul with vector x matrix', () => {
    const batch = 3;
    const sharedDim = MATMUL_SHARED_DIM_THRESHOLD + 1;
    const values = new Float32Array(batch * sharedDim);
    values[10] = 2;

    const a = tf.tensor(values, [batch, 1, sharedDim]);
    const b = tf.ones([batch, sharedDim, 2]);
    const result = tf.matMul(a, b);
    expect(result.shape).toEqual([batch, 1, 2]);
    expectArraysClose(result, [2, 2, 0, 0, 0, 0]);
  });

  it('batched matmul with vector x matrix transposedA', () => {
    const batch = 3;
    const sharedDim = MATMUL_SHARED_DIM_THRESHOLD + 1;
    const values = new Float32Array(batch * sharedDim);
    values[10] = 2;

    const a = tf.tensor(values, [batch, sharedDim, 1]);
    const b = tf.ones([batch, sharedDim, 2]);
    const transposeA = true;
    const transposeB = false;
    const result = tf.matMul(a, b, transposeA, transposeB);
    expect(result.shape).toEqual([batch, 1, 2]);
    expectArraysClose(result, [2, 2, 0, 0, 0, 0]);
  });

  it('batched matmul with vector x matrix transposedB', () => {
    const batch = 3;
    const sharedDim = MATMUL_SHARED_DIM_THRESHOLD + 1;
    const values = new Float32Array(batch * sharedDim);
    values[10] = 2;

    const a = tf.tensor(values, [batch, 1, sharedDim]);
    const b = tf.ones([batch, 2, sharedDim]);
    const transposeA = false;
    const transposeB = true;
    const result = tf.matMul(a, b, transposeA, transposeB);
    expect(result.shape).toEqual([batch, 1, 2]);
    expectArraysClose(result, [2, 2, 0, 0, 0, 0]);
  });

  it('Matrix * vector propagates NaNs', () => {
    const matrix = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const v = tf.tensor1d([2, NaN]);
    const result = tf.dot(matrix, v);

    const expected = [NaN, NaN];
    expectArraysClose(result, expected);
  });

  it('matrix times vector throws when not passed a matrix', () => {
    const v = tf.tensor1d([2, 3]);

    // tslint:disable-next-line:no-any
    const matrix: any = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);

    expect(() => tf.dot(matrix, v)).toThrowError();
  });

  it('Dot product', () => {
    const v1 = tf.tensor1d([2, 3]);
    const v2 = tf.tensor1d([2, 1]);
    const result = tf.dot(v1, v2);

    expectNumbersClose(result.get(), 7);
  });

  it('Dot product propagates NaNs', () => {
    const v1 = tf.tensor1d([2, NaN]);
    const v2 = tf.tensor1d([2, 1]);
    const result = tf.dot(v1, v2);
    expect(result.get()).toEqual(NaN);
  });

  it('Dot product throws when vectors are different size', () => {
    const v1 = tf.tensor1d([2, 3, 3]);
    const v2 = tf.tensor1d([2, 1]);

    expect(() => tf.dot(v1, v2)).toThrowError();
    expect(() => tf.dot(v2, v1)).toThrowError();
  });

  it('Outer product', () => {
    const v1 = tf.tensor1d([2, 3]);
    const v2 = tf.tensor1d([2, 1]);
    const result = tf.outerProduct(v1, v2);

    const expected = [4, 2, 6, 3];
    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(result, expected);
  });

  it('outer product accepts a tensor-like object', () => {
    const v1 = [2, 3];
    const v2 = [2, 1];
    const result = tf.outerProduct(v1, v2);
    const expected = [4, 2, 6, 3];
    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(result, expected);
  });

  it('gradients: A * B', () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, 30], [2, 3]);
    const b = tf.tensor2d([2, 3, 4, 1, 2, 3], [3, 2]);
    const dy = tf.tensor2d([1, 10, 20, 30], [2, 2]);

    const transposeA = false;
    const transposeB = false;
    const grads = tf.grads(

        (a: tf.Tensor2D, b: tf.Tensor2D) =>
            tf.matMul(a, b, transposeA, transposeB));
    const [da, db] = grads([a, b], dy);

    // da = dy * bT
    expect(da.shape).toEqual(a.shape);
    expectArraysClose(
        da,
        [
          dy.get(0, 0) * b.get(0, 0) + dy.get(0, 1) * b.get(0, 1),
          dy.get(0, 0) * b.get(1, 0) + dy.get(0, 1) * b.get(1, 1),
          dy.get(0, 0) * b.get(2, 0) + dy.get(0, 1) * b.get(2, 1),
          dy.get(1, 0) * b.get(0, 0) + dy.get(1, 1) * b.get(0, 1),
          dy.get(1, 0) * b.get(1, 0) + dy.get(1, 1) * b.get(1, 1),
          dy.get(1, 0) * b.get(2, 0) + dy.get(1, 1) * b.get(2, 1)
        ],
        1e-1);

    // db = aT * dy
    expect(db.shape).toEqual(b.shape);
    expectArraysClose(db, [
      a.get(0, 0) * dy.get(0, 0) + a.get(1, 0) * dy.get(1, 0),
      a.get(0, 0) * dy.get(0, 1) + a.get(1, 0) * dy.get(1, 1),
      a.get(0, 1) * dy.get(0, 0) + a.get(1, 1) * dy.get(1, 0),
      a.get(0, 1) * dy.get(0, 1) + a.get(1, 1) * dy.get(1, 1),
      a.get(0, 2) * dy.get(0, 0) + a.get(1, 2) * dy.get(1, 0),
      a.get(0, 2) * dy.get(0, 1) + a.get(1, 2) * dy.get(1, 1)
    ]);
  });

  it('gradients: a * bT', () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, 30], [3, 2]);
    const b = tf.tensor2d([2, 3, 4, 1, 2, 3], [3, 2]);
    const dy = tf.tensor2d([1, 10, 20, 30, 40, 50, 60, 70, 80], [3, 3]);

    const transposeA = false;
    const transposeB = true;
    const grads = tf.grads(

        (a: tf.Tensor2D, b: tf.Tensor2D) =>
            tf.matMul(a, b, transposeA, transposeB));
    const [da, db] = grads([a, b], dy);

    // da = dy * b
    expect(da.shape).toEqual(a.shape);
    expectArraysClose(da, [
      dy.get(0, 0) * b.get(0, 0) + dy.get(0, 1) * b.get(1, 0) +
          dy.get(0, 2) * b.get(2, 0),
      dy.get(0, 0) * b.get(0, 1) + dy.get(0, 1) * b.get(1, 1) +
          dy.get(0, 2) * b.get(2, 1),
      dy.get(1, 0) * b.get(0, 0) + dy.get(1, 1) * b.get(1, 0) +
          dy.get(1, 2) * b.get(2, 0),
      dy.get(1, 0) * b.get(0, 1) + dy.get(1, 1) * b.get(1, 1) +
          dy.get(1, 2) * b.get(2, 1),
      dy.get(2, 0) * b.get(0, 0) + dy.get(2, 1) * b.get(1, 0) +
          dy.get(2, 2) * b.get(2, 0),
      dy.get(2, 0) * b.get(0, 1) + dy.get(2, 1) * b.get(1, 1) +
          dy.get(2, 2) * b.get(2, 1)
    ]);

    // db = dyT * a
    expect(db.shape).toEqual(b.shape);
    expectArraysClose(db, [
      dy.get(0, 0) * a.get(0, 0) + dy.get(1, 0) * a.get(1, 0) +
          dy.get(2, 0) * a.get(2, 0),
      dy.get(0, 0) * a.get(0, 1) + dy.get(1, 0) * a.get(1, 1) +
          dy.get(2, 0) * a.get(2, 1),
      dy.get(0, 1) * a.get(0, 0) + dy.get(1, 1) * a.get(1, 0) +
          dy.get(2, 1) * a.get(2, 0),
      dy.get(0, 1) * a.get(0, 1) + dy.get(1, 1) * a.get(1, 1) +
          dy.get(2, 1) * a.get(2, 1),
      dy.get(0, 2) * a.get(0, 0) + dy.get(1, 2) * a.get(1, 0) +
          dy.get(2, 2) * a.get(2, 0),
      dy.get(0, 2) * a.get(0, 1) + dy.get(1, 2) * a.get(1, 1) +
          dy.get(2, 2) * a.get(2, 1)
    ]);
  });

  it('gradients: aT * b', () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, 30], [3, 2]);
    const b = tf.tensor2d([2, 3, 4, 1, 2, 3], [3, 2]);
    const dy = tf.tensor2d([1, 10, 20, 30], [2, 2]);

    const transposeA = true;
    const transposeB = false;
    const grads = tf.grads(

        (a: tf.Tensor2D, b: tf.Tensor2D) =>
            tf.matMul(a, b, transposeA, transposeB));
    const [da, db] = grads([a, b], dy);

    // da = b * dyT
    expect(da.shape).toEqual(a.shape);
    expectArraysClose(da, [
      dy.get(0, 0) * b.get(0, 0) + dy.get(0, 1) * b.get(0, 1),
      dy.get(1, 0) * b.get(0, 0) + dy.get(1, 1) * b.get(0, 1),
      dy.get(0, 0) * b.get(1, 0) + dy.get(0, 1) * b.get(1, 1),
      dy.get(1, 0) * b.get(1, 0) + dy.get(1, 1) * b.get(1, 1),
      dy.get(0, 0) * b.get(2, 0) + dy.get(0, 1) * b.get(2, 1),
      dy.get(1, 0) * b.get(2, 0) + dy.get(1, 1) * b.get(2, 1)
    ]);

    // db = a * dy
    expect(db.shape).toEqual(b.shape);
    expectArraysClose(db, [
      dy.get(0, 0) * a.get(0, 0) + dy.get(1, 0) * a.get(0, 1),
      dy.get(0, 1) * a.get(0, 0) + dy.get(1, 1) * a.get(0, 1),
      dy.get(0, 0) * a.get(1, 0) + dy.get(1, 0) * a.get(1, 1),
      dy.get(0, 1) * a.get(1, 0) + dy.get(1, 1) * a.get(1, 1),
      dy.get(0, 0) * a.get(2, 0) + dy.get(1, 0) * a.get(2, 1),
      dy.get(0, 1) * a.get(2, 0) + dy.get(1, 1) * a.get(2, 1)
    ]);
  });

  it('gradients: aT * bT', () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, 30], [3, 2]);
    const b = tf.tensor2d([2, 3, 4, 1, 2, 3], [2, 3]);
    const dy = tf.tensor2d([1, 10, 20, 30], [2, 2]);

    const transposeA = true;
    const transposeB = true;
    const grads = tf.grads(

        (a: tf.Tensor2D, b: tf.Tensor2D) =>
            tf.matMul(a, b, transposeA, transposeB));
    const [da, db] = grads([a, b], dy);

    // da = bT * dyT
    expect(da.shape).toEqual(a.shape);
    expectArraysClose(da, [
      dy.get(0, 0) * b.get(0, 0) + dy.get(0, 1) * b.get(1, 0),
      dy.get(1, 0) * b.get(0, 0) + dy.get(1, 1) * b.get(1, 0),
      dy.get(0, 0) * b.get(0, 1) + dy.get(0, 1) * b.get(1, 1),
      dy.get(1, 0) * b.get(0, 1) + dy.get(1, 1) * b.get(1, 1),
      dy.get(0, 0) * b.get(0, 2) + dy.get(0, 1) * b.get(1, 2),
      dy.get(1, 0) * b.get(0, 2) + dy.get(1, 1) * b.get(1, 2)
    ]);

    // db = dyT * aT
    expect(db.shape).toEqual(b.shape);
    expectArraysClose(db, [
      dy.get(0, 0) * a.get(0, 0) + dy.get(1, 0) * a.get(0, 1),
      dy.get(0, 0) * a.get(1, 0) + dy.get(1, 0) * a.get(1, 1),
      dy.get(0, 0) * a.get(2, 0) + dy.get(1, 0) * a.get(2, 1),
      dy.get(0, 1) * a.get(0, 0) + dy.get(1, 1) * a.get(0, 1),
      dy.get(0, 1) * a.get(1, 0) + dy.get(1, 1) * a.get(1, 1),
      dy.get(0, 1) * a.get(2, 0) + dy.get(1, 1) * a.get(2, 1)
    ]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.matMul({} as tf.Tensor2D, tf.tensor2d([2], [1, 1])))
        .toThrowError(/Argument 'a' passed to 'matMul' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.matMul(tf.tensor2d([2], [1, 1]), {} as tf.Tensor2D))
        .toThrowError(/Argument 'b' passed to 'matMul' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const a = [[1, 2, 3], [4, 5, 6]];     // 2x3
    const b = [[0, 1], [-3, 2], [2, 1]];  // 3x2
    const c = tf.matMul(a, b);

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(c, [0, 8, -3, 20]);
  });

  it('a * b where a has zero in its shape', () => {
    const a = tf.tensor2d([], [0, 3]);
    const b = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    const c = tf.matMul(a, b);
    expect(c.shape).toEqual([0, 2]);
    expect(c.rank).toBe(2);
    expect(c.size).toBe(0);
    expectArraysClose(c, []);
  });

  it('(a * b) * c where a has zero in its shape, so a*b does also', () => {
    const a = tf.tensor2d([], [0, 3]);
    const b = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    const ab = tf.matMul(a, b);
    expect(ab.shape).toEqual([0, 2]);
    expectArraysClose(ab, []);
    const c = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const res = tf.matMul(ab, c);
    expect(res.shape).toEqual([0, 3]);
    expectArraysClose(res, []);
  });
});

describeWithFlags('matmulBatch', ALL_ENVS, () => {
  it('A x B', () => {
    const a = tf.tensor3d(
        [
          -5, -5, -6, 8, -2, -8, 4, -7, -6, -9, -1, 3,  7,  -2, 5,
          -6, 3,  8,  7, -8, 1,  4, -4, 6,  4,  -4, -9, -5, 2,  -2
        ],
        [5, 2, 3]);
    const b = tf.tensor3d(
        [
          -8, -4, -1, 0,  -7, 0, 3,  3,  6,   2,  -1, 8, -4, 9, -6,
          5,  8,  9,  -9, 7,  0, -1, -1, -10, -7, 3,  4, 6,  3, -4
        ],
        [5, 3, 2]);

    const c = tf.matMul(a, b);
    expect(c.shape).toEqual([5, 2, 2]);
    expectArraysClose(c, [
      87, 20, -6,  -32, -24, -50, -36, -5, 24, 98,
      70, 33, -64, 47,  -42, -28, -71, 24, 37, 5
    ]);
  });

  it('A x B in 4D', () => {
    const a = tf.tensor4d(
        [
          -2, 3,  5,  -5, 3,  9,  -3, -5, 1,   1,  -9, 9,   -6, 6,  -8,
          -7, -1, 3,  9,  -7, -7, 2,  10, -6,  -8, -6, 9,   -6, 4,  -1,
          9,  -6, 10, 8,  -9, 5,  -8, -7, 0,   2,  -5, -1,  -9, -4, 3,
          -2, 6,  -4, 7,  1,  -5, -4, 9,  -8,  -6, -8, 4,   -1, 4,  3,
          -7, 8,  -7, 5,  -3, -2, -4, 9,  2,   -1, 1,  -10, -3, 5,  -4,
          6,  -8, -8, 9,  -3, -5, 10, 3,  -3,  -3, 9,  3,   -3, 2,  -8,
          10, 1,  9,  -2, -2, -3, -4, 6,  -10, -1, 8,  -8,  7,  3,  -2,
          3,  6,  -2, -2, -4, 1,  -5, -4, 0,   5,  1,  9,   -8, -2, -1
        ],
        [4, 5, 2, 3]);
    const b = tf.tensor4d(
        [
          -4, -3, -2, -6, 6,  -1, -4, -1, 7,  -4, 8,  -9,  -9, 0,   -1,
          -4, -6, -7, -3, -4, -7, 6,  -8, 1,  -2, 1,  -1,  -3, 8,   -5,
          9,  -2, 5,  9,  -2, 2,  -5, -5, -8, -1, -2, -3,  -2, -10, 6,
          -3, 0,  1,  6,  7,  1,  2,  -4, -5, 2,  -5, -7,  9,  3,   -6,
          6,  4,  -4, 6,  10, -3, -2, 8,  10, -8, 10, -1,  -9, -7,  -8,
          -3, 1,  1,  -2, -9, -7, -6, -1, 0,  7,  -9, -7,  -5, 0,   -4,
          -4, -7, 2,  4,  6,  6,  -4, -6, -8, 3,  -8, -9,  6,  9,   -4,
          1,  -1, 0,  8,  9,  0,  -5, 3,  -1, 5,  0,  -10, 7,  -2,  6
        ],
        [4, 5, 3, 2]);

    const transposeA = false;
    const transposeB = false;
    const c = tf.matMul(a, b, transposeA, transposeB);

    expectArraysClose(c, [
      32,  -17, 68,  -12,  -15, 14,  5,   -46, 96,  32,  46,  -17, 78,   -85,
      -28, 46,  94,  -35,  0,   -13, 31,  -52, 17,  -87, 96,  47,  32,   -2,
      -6,  105, 40,  -2,   63,  76,  17,  30,  56,  -66, -21, 23,  -144, 41,
      22,  8,   118, -106, -88, -6,  -17, 2,   2,   -26, 8,   -63, -38,  -108,
      -84, -30, -35, 49,   16,  -12, -14, -12, 48,  132, 4,   102, 32,   66,
      -4,  33,  -13, 1,    -40, -25, -3,  61,  -18, -20
    ]);
  });

  it('A x B^t', () => {
    const a = tf.tensor3d(
        [
          -5, -5, -6, 8, -2, -8, 4, -7, -6, -9, -1, 3,  7,  -2, 5,
          -6, 3,  8,  7, -8, 1,  4, -4, 6,  4,  -4, -9, -5, 2,  -2
        ],
        [5, 2, 3]);
    const b = tf.tensor3d(
        [
          -8, -4, -1, 0,  -7, 0, 3,  3,  6,   2,  -1, 8, -4, 9, -6,
          5,  8,  9,  -9, 7,  0, -1, -1, -10, -7, 3,  4, 6,  3, -4
        ],
        [5, 2, 3]);

    const transposeA = false;
    const transposeB = true;
    const c = tf.matMul(a, b, transposeA, transposeB);
    expect(c.shape).toEqual([5, 2, 2]);
    expectArraysClose(c, [
      66, 35, -48,  14, -45, -33, -12, 7,  -76, 64,
      3,  66, -119, -9, -64, -60, -76, 48, 33,  -16
    ]);
  });

  it('A^t x B', () => {
    const a = tf.tensor3d(
        [
          -5, -5, -6, 8, -2, -8, 4, -7, -6, -9, -1, 3,  7,  -2, 5,
          -6, 3,  8,  7, -8, 1,  4, -4, 6,  4,  -4, -9, -5, 2,  -2
        ],
        [5, 2, 3]);
    const b = tf.tensor3d(
        [
          -8, -4, -1, 0,  -7, 0, 3,  3,  6,   2,  -1, 8, -4, 9, -6,
          5,  8,  9,  -9, 7,  0, -1, -1, -10, -7, 3,  4, 6,  3, -4
        ],
        [5, 2, 3]);

    const transposeA = true;
    const transposeB = false;
    const c = tf.matMul(a, b, transposeA, transposeB);

    expectArraysClose(c, [
      40,  -36, 5,   40,  34, 5,   48,  80, 6,  -6, 21,  -48, -23, -20, -50,
      -12, -21, -12, -58, 15, -96, 23,  6,  39, 20, 109, 42,  -67, 45,  -40,
      76,  -52, 40,  -15, 1,  -60, -58, -3, 36, 40, -6,  -24, 51,  -33, -28
    ]);
  });

  it('A^t x B in 4D', () => {
    const a = tf.tensor4d(
        [
          -2, 3,  5,  -5, 3,  9,  -3, -5, 1,   1,  -9, 9,   -6, 6,  -8,
          -7, -1, 3,  9,  -7, -7, 2,  10, -6,  -8, -6, 9,   -6, 4,  -1,
          9,  -6, 10, 8,  -9, 5,  -8, -7, 0,   2,  -5, -1,  -9, -4, 3,
          -2, 6,  -4, 7,  1,  -5, -4, 9,  -8,  -6, -8, 4,   -1, 4,  3,
          -7, 8,  -7, 5,  -3, -2, -4, 9,  2,   -1, 1,  -10, -3, 5,  -4,
          6,  -8, -8, 9,  -3, -5, 10, 3,  -3,  -3, 9,  3,   -3, 2,  -8,
          10, 1,  9,  -2, -2, -3, -4, 6,  -10, -1, 8,  -8,  7,  3,  -2,
          3,  6,  -2, -2, -4, 1,  -5, -4, 0,   5,  1,  9,   -8, -2, -1
        ],
        [4, 5, 2, 3]);
    const b = tf.tensor4d(
        [
          -4, -3, -2, -6, 6,  -1, -4, -1, 7,  -4, 8,  -9,  -9, 0,   -1,
          -4, -6, -7, -3, -4, -7, 6,  -8, 1,  -2, 1,  -1,  -3, 8,   -5,
          9,  -2, 5,  9,  -2, 2,  -5, -5, -8, -1, -2, -3,  -2, -10, 6,
          -3, 0,  1,  6,  7,  1,  2,  -4, -5, 2,  -5, -7,  9,  3,   -6,
          6,  4,  -4, 6,  10, -3, -2, 8,  10, -8, 10, -1,  -9, -7,  -8,
          -3, 1,  1,  -2, -9, -7, -6, -1, 0,  7,  -9, -7,  -5, 0,   -4,
          -4, -7, 2,  4,  6,  6,  -4, -6, -8, 3,  -8, -9,  6,  9,   -4,
          1,  -1, 0,  8,  9,  0,  -5, 3,  -1, 5,  0,  -10, 7,  -2,  6
        ],
        [4, 5, 2, 3]);

    const transposeA = true;
    const transposeB = false;
    const c = tf.matMul(a, b, transposeA, transposeB);

    expectArraysClose(c, [
      38,  -24, 9,   -30, 9,   -9,  -74,  39,  -19,  8,    11,  -30, 56,  -67,
      46,  -40, 71,  -74, 82,  42,  55,   -50, 6,    1,    60,  -18, -13, -15,
      -52, -61, 81,  -52, 59,  -15, 76,   43,  34,   -56,  38,  0,   26,  -14,
      -15, 1,   -4,  153, -34, 61,  -135, 30,  -48,  135,  -30, 60,  38,  36,
      58,  40,  45,  71,  1,   2,   3,    24,  90,   -56,  -10, 40,  -18, 6,
      -30, 14,  34,  65,  27,  24,  -29,  -44, -46,  -3,   35,  -21, 27,  48,
      20,  52,  32,  35,  -11, -46, -12,  22,  13,   30,   2,   -23, -54, -48,
      34,  16,  -42, -39, -26, 82,  89,   76,  -84,  30,   9,   27,  30,  -21,
      -43, -48, 60,  20,  24,  -78, -91,  -63, -12,  24,   21,  28,  48,  35,
      -6,  27,  33,  53,  -81, -71, 61,   -27, 11,   -48,  -82, 8,   -12, -19,
      -10, -48, -81, 0,   13,  32,  41,   0,   -100, -120, 16,  124, 152, 45,
      60,  -28, 24,  21,  -12, -14, -16,  8,   9,    -33,  5,   -12, -48, 4,
      8,   9,   0,   -31, 16,  -98, -9,   4,   -22,  38,   2,   -96
    ]);
  });

  it('A^t x B^t', () => {
    const a = tf.tensor3d(
        [
          -5, -5, -6, 8, -2, -8, 4, -7, -6, -9, -1, 3,  7,  -2, 5,
          -6, 3,  8,  7, -8, 1,  4, -4, 6,  4,  -4, -9, -5, 2,  -2
        ],
        [5, 3, 2]);
    const b = tf.tensor3d(
        [
          -8, -4, -1, 0,  -7, 0, 3,  3,  6,   2,  -1, 8, -4, 9, -6,
          5,  8,  9,  -9, 7,  0, -1, -1, -10, -7, 3,  4, 6,  3, -4
        ],
        [5, 2, 3]);

    const transposeA = true;
    const transposeB = true;
    const c = tf.matMul(a, b, transposeA, transposeB);
    expectArraysClose(c, [
      66,  42, 16,  -56, -12, 6,   -30, 19,  -1, 102,
      -94, 14, -56, 32,  100, -56, -47, -11, 5,  -31
    ]);
  });

  it('batch dimensions do not match', () => {
    const a = tf.tensor3d(
        [
          -5, -5, -6, 8,  -2, -8, 4, -7, -6, -9, -1, 3,
          7,  -2, 5,  -6, 3,  8,  7, -8, 1,  4,  -4, 6
        ],
        [4, 3, 2]);
    const b = tf.tensor3d(
        [
          -8, -4, -1, 0,  -7, 0, 3,  3,  6,   2,  -1, 8, -4, 9, -6,
          5,  8,  9,  -9, 7,  0, -1, -1, -10, -7, 3,  4, 6,  3, -4
        ],
        [5, 2, 3]);

    const f = () => {
      tf.matMul(a, b, false, false);
    };
    expect(f).toThrowError();
  });

  it('gradients: A x B', () => {
    const a = tf.tensor3d(
        [
          -5, -5, -6, 8, -2, -8, 4, -7, -6, -9, -1, 3,  7,  -2, 5,
          -6, 3,  8,  7, -8, 1,  4, -4, 6,  4,  -4, -9, -5, 2,  -2
        ],
        [5, 2, 3]);
    const b = tf.tensor3d(
        [
          -8, -4, -1, 0,  -7, 0, 3,  3,  6,   2,  -1, 8, -4, 9, -6,
          5,  8,  9,  -9, 7,  0, -1, -1, -10, -7, 3,  4, 6,  3, -4
        ],
        [5, 3, 2]);
    const dy = tf.tensor3d(
        [8, 2, -3, -2, -8, 4, 5, 7, 4, -4, -4, 5, 8, 10, 1, 0, 6, 6, -4, 7],
        [5, 2, 2]);

    const grads = tf.grads(
        (a: tf.Tensor3D, b: tf.Tensor3D) => tf.matMul(a, b, false, false));
    const [da, db] = grads([a, b], dy);

    // da = dy * bT
    expect(da.shape).toEqual(a.shape);
    expectArraysClose(da, [
      -72, -8, -56, 32, 3,   21,   -12, -40, 40, 36,  44, 51, -52, -44, -4,
      61,  49, 13,  -2, -10, -108, -9,  0,   -1, -24, 60, -6, 49,  26,  -40
    ]);

    // db = aT * dy
    expect(db.shape).toEqual(b.shape);
    expectArraysClose(db, [
      -64, -26, -34, -6, -24, 4,   -77, -47, 51, -35, 63,  -3,  52,  -58, -20,
      23,  -12, 20,  60, 70,  -68, -80, 14,  10, 44,  -11, -32, -10, -46, -68
    ]);
  });

  it('4d gradients: A x B', () => {
    const a = tf.tensor4d(
        [
          -2, 3,  5,  -5, 3,  9,  -3, -5, 1,   1,  -9, 9,   -6, 6,  -8,
          -7, -1, 3,  9,  -7, -7, 2,  10, -6,  -8, -6, 9,   -6, 4,  -1,
          9,  -6, 10, 8,  -9, 5,  -8, -7, 0,   2,  -5, -1,  -9, -4, 3,
          -2, 6,  -4, 7,  1,  -5, -4, 9,  -8,  -6, -8, 4,   -1, 4,  3,
          -7, 8,  -7, 5,  -3, -2, -4, 9,  2,   -1, 1,  -10, -3, 5,  -4,
          6,  -8, -8, 9,  -3, -5, 10, 3,  -3,  -3, 9,  3,   -3, 2,  -8,
          10, 1,  9,  -2, -2, -3, -4, 6,  -10, -1, 8,  -8,  7,  3,  -2,
          3,  6,  -2, -2, -4, 1,  -5, -4, 0,   5,  1,  9,   -8, -2, -1
        ],
        [4, 5, 2, 3]);
    const b = tf.tensor4d(
        [
          -4, -3, -2, -6, 6,  -1, -4, -1, 7,  -4, 8,  -9,  -9, 0,   -1,
          -4, -6, -7, -3, -4, -7, 6,  -8, 1,  -2, 1,  -1,  -3, 8,   -5,
          9,  -2, 5,  9,  -2, 2,  -5, -5, -8, -1, -2, -3,  -2, -10, 6,
          -3, 0,  1,  6,  7,  1,  2,  -4, -5, 2,  -5, -7,  9,  3,   -6,
          6,  4,  -4, 6,  10, -3, -2, 8,  10, -8, 10, -1,  -9, -7,  -8,
          -3, 1,  1,  -2, -9, -7, -6, -1, 0,  7,  -9, -7,  -5, 0,   -4,
          -4, -7, 2,  4,  6,  6,  -4, -6, -8, 3,  -8, -9,  6,  9,   -4,
          1,  -1, 0,  8,  9,  0,  -5, 3,  -1, 5,  0,  -10, 7,  -2,  6
        ],
        [4, 5, 3, 2]);
    const dy = tf.tensor4d(
        [
          8,  -7, 0,  -9,  -5, -5, 0,  3,  7,  -4, 6,  -8,  -8, 0,  -1, -8,
          -9, -7, -4, -9,  2,  3,  5,  8,  -5, -7, 3,  -10, -5, -9, -5, 1,
          7,  1,  -9, -10, 8,  5,  0,  8,  -6, 4,  0,  -5,  8,  -7, -2, 1,
          -8, 9,  9,  -7,  1,  7,  -2, 5,  -2, 9,  1,  -5,  7,  5,  -7, -6,
          6,  7,  -8, 7,   4,  -5, 4,  -5, 3,  -4, -5, 4,   -6, 3,  -8, 10
        ],
        [4, 5, 2, 2]);

    const grads = tf.grads(
        (a: tf.Tensor4D, b: tf.Tensor4D) => tf.matMul(a, b, false, false));
    const [da, db] = grads([a, b], dy);

    // da = dy * bT
    expect(da.shape).toEqual(a.shape);
    expectArraysClose(da, [
      -11,  26,  55,  27,  54,  9,   25,  -15, 5,   -3,   -12, -27, -63, 9,
      -14,  -54, 26,  20,  24,  56,  64,  35,  -41, 0,    11,  30,  -37, -1,
      31,   13,  12,  37,  2,   29,  97,  6,   60,  47,   31,  35,  -14, 24,
      100,  -3,  -9,  0,   -33, 1,   49,  9,   -33, -124, -29, 86,  -9,  -11,
      -6,   -40, 72,  -48, -20, 48,  -72, -20, -30, 15,   -72, 136, 87,  12,
      -28,  -21, 9,   37,  1,   -32, -51, 2,   -65, -49,  -1,  -41, -16, 2,
      -95,  -31, -36, 52,  18,  20,  -63, 34,  72,  70,   -38, -78, -66, -27,
      -111, -10, 85,  1,   -21, -21, -4,  -21, -21, -4,   -12, 20,  13,  -4,
      -20,  -19, -30, 81,  30,  -40, 150, 76
    ]);

    // db = aT * dy
    expect(db.shape).toEqual(b.shape);
    expectArraysClose(db, [
      -16, 59,  24,  -48,  40,   -116, 15,  18,  25,  -2,  -5,  22,  -84, 80,
      36,  -16, -38, 8,    -74,  -16,  46,  -80, 62,  48,  96,  110, 38,  6,
      -77, -54, 58,  91,   -57,  -90,  45,  70,  46,  36,  20,  99,  -3,  10,
      55,  79,  -10, 42,   5,    -31,  85,  47,  -74, -89, 37,  75,  -48, -38,
      -64, -8,  32,  44,   42,   -53,  -48, 47,  42,  -18, -30, 27,  70,  -62,
      36,  -24, 78,  -69,  -112, 101,  -40, 20,  -11, 113, -9,  -6,  1,   -50,
      3,   -12, -16, 71,   -14,  67,   84,  62,  21,  17,  84,  63,  -16, -35,
      -28, 98,  4,   -126, 40,   -50,  36,  -45, -16, 20,  19,  -12, 8,   0,
      3,   -4,  34,  -65,  10,   -17,  -46, 17
    ]);
  });

  it('gradients: A x B^t', () => {
    const a = tf.tensor3d(
        [
          -5, -5, -6, 8, -2, -8, 4, -7, -6, -9, -1, 3,  7,  -2, 5,
          -6, 3,  8,  7, -8, 1,  4, -4, 6,  4,  -4, -9, -5, 2,  -2
        ],
        [5, 3, 2]);
    const b = tf.tensor3d(
        [
          -8, -4, -1, 0,  -7, 0, 3,  3,  6,   2,  -1, 8, -4, 9, -6,
          5,  8,  9,  -9, 7,  0, -1, -1, -10, -7, 3,  4, 6,  3, -4
        ],
        [5, 3, 2]);
    const dy = tf.tensor3d(
        [
          -0, 7,  5, 0,  -9, 5, -7, 6,  -5, -3,  -2, -2, -4, 10, -3,
          5,  -1, 3, -2, -9, 4, -5, 7,  9,  -10, -8, -8, -5, -0, -1,
          3,  3,  4, 9,  -7, 6, -2, -9, 5,  1,   -5, -3, -1, 9,  4
        ],
        [5, 3, 3]);

    const grads = tf.grads(
        (a: tf.Tensor3D, b: tf.Tensor3D) => tf.matMul(a, b, false, true));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expectArraysClose(da, [
      -42, 0,  -26,  0,  85,  28,  -19, -29, 51, -16, 6,   37,  94,  -27, 50,
      71,  24, -202, 46, -25, -31, -22, -87, 10, -7,  -80, -36, -15, 55,  35
    ]);

    expect(db.shape).toEqual(b.shape);
    expectArraysClose(db, [
      14,  56, 7,    -155, -45, 55, 7,   72,  -67, -79, 7, 50, -69, -46, -52,
      -88, 49, -126, -68,  106, 31, -30, -27, 60,  -19, 5, 27, 43,  55,  -13
    ]);
  });

  it('4d gradients: A x B^t', () => {
    const a = tf.tensor4d(
        [
          -2, 3,  5,  -5, 3,  9,  -3, -5, 1,   1,  -9, 9,   -6, 6,  -8,
          -7, -1, 3,  9,  -7, -7, 2,  10, -6,  -8, -6, 9,   -6, 4,  -1,
          9,  -6, 10, 8,  -9, 5,  -8, -7, 0,   2,  -5, -1,  -9, -4, 3,
          -2, 6,  -4, 7,  1,  -5, -4, 9,  -8,  -6, -8, 4,   -1, 4,  3,
          -7, 8,  -7, 5,  -3, -2, -4, 9,  2,   -1, 1,  -10, -3, 5,  -4,
          6,  -8, -8, 9,  -3, -5, 10, 3,  -3,  -3, 9,  3,   -3, 2,  -8,
          10, 1,  9,  -2, -2, -3, -4, 6,  -10, -1, 8,  -8,  7,  3,  -2,
          3,  6,  -2, -2, -4, 1,  -5, -4, 0,   5,  1,  9,   -8, -2, -1
        ],
        [4, 5, 3, 2]);
    const b = tf.tensor4d(
        [
          -4, -3, -2, -6, 6,  -1, -4, -1, 7,  -4, 8,  -9,  -9, 0,   -1,
          -4, -6, -7, -3, -4, -7, 6,  -8, 1,  -2, 1,  -1,  -3, 8,   -5,
          9,  -2, 5,  9,  -2, 2,  -5, -5, -8, -1, -2, -3,  -2, -10, 6,
          -3, 0,  1,  6,  7,  1,  2,  -4, -5, 2,  -5, -7,  9,  3,   -6,
          6,  4,  -4, 6,  10, -3, -2, 8,  10, -8, 10, -1,  -9, -7,  -8,
          -3, 1,  1,  -2, -9, -7, -6, -1, 0,  7,  -9, -7,  -5, 0,   -4,
          -4, -7, 2,  4,  6,  6,  -4, -6, -8, 3,  -8, -9,  6,  9,   -4,
          1,  -1, 0,  8,  9,  0,  -5, 3,  -1, 5,  0,  -10, 7,  -2,  6
        ],
        [4, 5, 3, 2]);
    const dy = tf.tensor4d(
        [
          5,  -1, -5, -4, -1,  9,  1,   -2, 10,  7,  -1, 6,   -8, 8,  -3,
          9,  -4, 2,  -4, -8,  8,  4,   8,  -10, -8, -8, 6,   6,  -5, 9,
          -1, -7, -5, -3, -3,  2,  -6,  5,  8,   -9, 5,  -8,  -3, 8,  6,
          2,  8,  5,  9,  7,   6,  2,   -3, 10,  7,  7,  -3,  4,  -3, -6,
          -8, -8, 9,  0,  -8,  -3, -2,  -2, 8,   2,  3,  -6,  3,  6,  -3,
          7,  7,  -9, -3, 8,   7,  7,   -1, -6,  5,  2,  -1,  -1, 1,  5,
          0,  -4, 3,  -4, -10, 1,  -2,  -8, -9,  -6, 4,  4,   -7, -1, -1,
          -9, 7,  1,  -1, 8,   0,  -2,  -7, 5,   7,  8,  9,   -3, -8, -6,
          -7, -8, -1, 8,  -4,  7,  5,   -9, 9,   3,  0,  -10, 7,  -9, 4,
          -7, 5,  -2, -2, 3,   3,  -6,  2,  0,   8,  -5, -10, 3,  -7, 0,
          -6, 2,  3,  -1, 3,   3,  -10, 1,  3,   -7, -1, 8,   -2, -1, -1,
          -3, -9, 7,  4,  -6,  3,  0,   -7, -4,  -5, -8, -6,  10, -6, 4
        ],
        [4, 5, 3, 3]);

    const grads = tf.grads(
        (a: tf.Tensor4D, b: tf.Tensor4D) => tf.matMul(a, b, false, true));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expectArraysClose(da, [
      -48, -4,   72,  9,   60,  -1,  13,   -57, 64,  3,   -48, -11, -4,  -24,
      16,  38,   44,  -10, -55, -45, 92,   -43, 14,  -4,  71,  -61, -51, 16,
      46,  -57,  48,  78,  104, 57,  -17,  -11, -85, -33, 16,  1,   86,  21,
      -48, 21,   -8,  34,  14,  -35, 36,   48,  85,  108, -38, -40, 3,   -8,
      -7,  -1,   6,   -16, 46,  -33, 26,   -79, -70, -29, 92,  -84, -6,  -47,
      98,  -129, -55, -17, 79,  40,  -118, -64, 68,  75,  71,  111, 5,   -48,
      98,  -36,  21,  13,  112, -34, 26,   57,  32,  44,  28,  50,  88,  27,
      44,  -39,  -16, 15,  -21, -6,  -67,  -89, -46, -64, -19, -12, -3,  11,
      41,  63,   78,  -73, 67,  -92, 102,  -18
    ]);

    expect(db.shape).toEqual(b.shape);
    expectArraysClose(db, [
      -27,  44,   -9,  -16, 85,  30,  -110, 38,  47,  -23, -39, -15, 0,    -76,
      -8,   -128, 26,  136, 31,  -26, -26,  39,  136, -85, -45, 93,  37,   -68,
      -112, -6,   90,  70,  169, -7,  15,   68,  -16, -33, -16, -47, -21,  0,
      6,    -4,   84,  24,  15,  20,  -41,  -1,  79,  -86, 87,  -23, -26,  -64,
      18,   9,    52,  64,  34,  -16, 122,  -66, -1,  47,  1,   43,  -11,  -33,
      -17,  27,   -45, -73, -60, -66, -92,  -42, 32,  -85, -44, -44, -28,  -13,
      8,    -20,  9,   -9,  -49, 79,  -76,  15,  73,  -7,  7,   -8,  -110, 93,
      106,  -39,  64,  -84, -29, -19, 13,   14,  63,  2,   -15, 23,  17,   49,
      -3,   -31,  -65, 30,  -95, 63,  -82,  40
    ]);
  });

  it('gradients: A^t x B', () => {
    const a = tf.tensor3d(
        [
          -5, -5, -6, 8, -2, -8, 4, -7, -6, -9, -1, 3,  7,  -2, 5,
          -6, 3,  8,  7, -8, 1,  4, -4, 6,  4,  -4, -9, -5, 2,  -2
        ],
        [5, 3, 2]);
    const b = tf.tensor3d(
        [
          -8, -4, -1, 0,  -7, 0, 3,  3,  6,   2,  -1, 8, -4, 9, -6,
          5,  8,  9,  -9, 7,  0, -1, -1, -10, -7, 3,  4, 6,  3, -4
        ],
        [5, 3, 2]);
    const dy = tf.tensor3d(
        [8, 2, -3, -2, -8, 4, 5, 7, 4, -4, -4, 5, 8, 10, 1, 0, 6, 6, -4, 7],
        [5, 2, 2]);

    const grads = tf.grads(
        (a: tf.Tensor3D, b: tf.Tensor3D) => tf.matMul(a, b, true, false));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expectArraysClose(da, [
      -72, 32, -8, 3,  -56, 21,  -12, 36,   -40, 44,  40, 51, -52, 61, -44,
      49,  -4, 13, -2, -9,  -10, 0,   -108, -1,  -24, 49, 60, 26,  -6, -40
    ]);

    expect(db.shape).toEqual(b.shape);
    expectArraysClose(db, [
      -25, 0,   -72, -28, 8,  12, -67, -33, 3,   -87, 23, 17,  36,  -38, 44,
      -50, -20, 28,  48,  70, 12, 10,  -26, -40, 40,  -4, -34, -89, 20,  -2
    ]);
  });

  it('gradients: A^t x B^t', () => {
    const a = tf.tensor3d(
        [
          -5, -5, -6, 8, -2, -8, 4, -7, -6, -9, -1, 3,  7,  -2, 5,
          -6, 3,  8,  7, -8, 1,  4, -4, 6,  4,  -4, -9, -5, 2,  -2
        ],
        [5, 3, 2]);
    const b = tf.tensor3d(
        [
          -8, -4, -1, 0,  -7, 0, 3,  3,  6,   2,  -1, 8, -4, 9, -6,
          5,  8,  9,  -9, 7,  0, -1, -1, -10, -7, 3,  4, 6,  3, -4
        ],
        [5, 2, 3]);
    const dy = tf.tensor3d(
        [8, 2, -3, -2, -8, 4, 5, 7, 4, -4, -4, 5, 8, 10, 1, 0, 6, 6, -4, 7],
        [5, 2, 2]);

    const grads = tf.grads(
        (a: tf.Tensor3D, b: tf.Tensor3D) => tf.matMul(a, b, true, true));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expectArraysClose(da, [
      -64, 24,  -46, 26,  -8, 3,  -16, 29,   -28, 8,  -16, 86, -36, 41, 4,
      4,   -60, 69,  -82, -9, 46, 7,   -100, 0,   -6, 70,  36, 9,   0,  -44
    ]);

    expect(db.shape).toEqual(b.shape);
    expectArraysClose(db, [
      -25, -72, 8,  0,  -28, 12,  -67, 3,  23,  -33, -87, 17, 36, 44,  -20,
      -38, -50, 28, 48, 12,  -26, 70,  10, -40, 40,  -34, 20, -4, -89, -2
    ]);
  });
});

describeWithFlags('matmul webgl-only', WEBGL_ENVS, () => {
  it('Matrix times vector, large matrix', () => {
    const maxTexSize = 16000;
    const sharedDim = maxTexSize + 4;
    const matrix = tf.buffer<Rank.R2>([2, sharedDim], 'float32');
    matrix.set(1, 0, sharedDim - 3);
    matrix.set(1, 0, sharedDim - 2);

    const v = tf.buffer<Rank.R1>([sharedDim], 'float32');
    v.set(1, sharedDim - 3);
    v.set(1, sharedDim - 2);

    const result = tf.dot(matrix.toTensor(), v.toTensor());
    const expected = [2, 0];
    expectArraysClose(result, expected);
  });
});

describeWithFlags('dot', ALL_ENVS, () => {
  let a: tf.Tensor1D;
  let b: tf.Tensor2D;
  let c: tf.Tensor2D;
  let d: tf.Tensor3D;
  let e: tf.Scalar;

  beforeEach(() => {
    a = tf.tensor1d([1, 2]);
    b = tf.tensor2d([[1, 2], [3, 4]]);
    c = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
    d = tf.tensor3d([1, 2], [1, 1, 2]);
    e = tf.scalar(1);
  });

  it('vector-vector', () => {
    const aa = tf.dot(a, a);
    expectArraysClose(aa, [5]);
    expect(aa.shape).toEqual([]);
  });

  it('vector-matrix', () => {
    const ab = tf.dot(a, b);
    const ac = tf.dot(a, c);
    expect(ab.shape).toEqual([2]);
    expect(ac.shape).toEqual([3]);
    expectArraysClose(ab, [7, 10]);
    expectArraysClose(ac, [9, 12, 15]);
  });

  it('matrix-vector', () => {
    const ba = b.dot(a);
    expect(ba.shape).toEqual([2]);
    expectArraysClose(ba, [5, 11]);
  });

  it('matrix-matrix', () => {
    const bb = tf.dot(b, b);
    const bc = tf.dot(b, c);
    expect(bb.shape).toEqual([2, 2]);
    expect(bc.shape).toEqual([2, 3]);
    expectArraysClose(bb, [7, 10, 15, 22]);
    expectArraysClose(bc, [9, 12, 15, 19, 26, 33]);
  });

  it('throws error on incompatible dimensions', () => {
    expect(() => tf.dot(c, a)).toThrowError();
    expect(() => tf.dot(c, b)).toThrowError();
  });

  it('throws error when inputs are not rank 1 or 2', () => {
    expect(() => tf.dot(a, d)).toThrowError();
    expect(() => tf.dot(a, e)).toThrowError();
  });

  it('accepts a tensor-like object', () => {
    const a = [1, 2, 3];
    const res = tf.dot(a, a);
    expectArraysClose(res, [14]);
    expect(res.shape).toEqual([]);
  });
});
