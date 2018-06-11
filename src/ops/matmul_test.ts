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
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, expectArraysClose, expectNumbersClose, WEBGL_ENVS} from '../test_util';
import {Rank} from '../types';
import {MatmulOps} from './matmul';

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

  it('Vector times matrix', () => {
    const v = tf.tensor1d([2, 3]);
    const matrix = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const result = tf.vectorTimesMatrix(v, matrix);

    const expected = [11, 16];
    expectArraysClose(result, expected);
  });

  it('Vector times matrix with implicit reshape', () => {
    const v = tf.tensor1d([2, 3]);

    const matrix = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const result = tf.vectorTimesMatrix(v, matrix);

    const expected = [11, 16];
    expectArraysClose(result, expected);
  });

  it('Vector times matrix throws when not passed a vector', () => {
    // tslint:disable-next-line:no-any
    const v: any = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const matrix = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    expect(() => tf.vectorTimesMatrix(v, matrix)).toThrowError();
  });

  it('Vector times matrix throws when not passed a matrix', () => {
    const v = tf.tensor1d([2, 3]);
    // tslint:disable-next-line:no-any
    const matrix: any = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);

    expect(() => tf.vectorTimesMatrix(v, matrix)).toThrowError();
  });

  it('Matrix times vector', () => {
    const matrix = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const v = tf.tensor1d([2, 3]);
    const result = tf.matrixTimesVector(matrix, v);

    const expected = [8, 18];
    expectArraysClose(result, expected);
  });

  it('Matrix * vector propagates NaNs', () => {
    const matrix = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const v = tf.tensor1d([2, NaN]);
    const result = tf.matrixTimesVector(matrix, v);

    const expected = [NaN, NaN];
    expectArraysClose(result, expected);
  });

  it('matrix times vector throws when not passed a vector', () => {
    // tslint:disable-next-line:no-any
    const v: any = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const matrix = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    expect(() => tf.matrixTimesVector(matrix, v)).toThrowError();
  });

  it('matrix times vector throws when not passed a matrix', () => {
    const v = tf.tensor1d([2, 3]);

    // tslint:disable-next-line:no-any
    const matrix: any = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);

    expect(() => tf.matrixTimesVector(matrix, v)).toThrowError();
  });

  it('Dot product', () => {
    const v1 = tf.tensor1d([2, 3]);
    const v2 = tf.tensor1d([2, 1]);
    const result = MatmulOps.dotProduct(v1, v2);

    expectNumbersClose(result.get(), 7);
  });

  it('Dot product propagates NaNs', () => {
    const v1 = tf.tensor1d([2, NaN]);
    const v2 = tf.tensor1d([2, 1]);
    const result = MatmulOps.dotProduct(v1, v2);
    expect(result.get()).toEqual(NaN);
  });

  it('Dot product throws when vectors are different size', () => {
    const v1 = tf.tensor1d([2, 3, 3]);
    const v2 = tf.tensor1d([2, 1]);

    expect(() => MatmulOps.dotProduct(v1, v2)).toThrowError();
    expect(() => MatmulOps.dotProduct(v2, v1)).toThrowError();
  });

  it('Dot product throws when passed non vectors', () => {
    // tslint:disable-next-line:no-any
    const v1: any = tf.tensor2d([1, 2, 3, 3], [2, 2]);
    const v2 = tf.tensor1d([2, 1]);

    expect(() => MatmulOps.dotProduct(v1, v2)).toThrowError();
    expect(() => MatmulOps.dotProduct(v2, v1)).toThrowError();
  });

  it('Outer product', () => {
    const v1 = tf.tensor1d([2, 3]);
    const v2 = tf.tensor1d([2, 1]);
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

    const result = tf.matrixTimesVector(matrix.toTensor(), v.toTensor());
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
});
