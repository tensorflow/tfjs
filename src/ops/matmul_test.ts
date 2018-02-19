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

import * as dl from '../index';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, describeWithFlags, expectArraysClose, expectNumbersClose, WEBGL_ENVS} from '../test_util';
import {Rank} from '../types';
import {Ops as MatmulOps} from './matmul';

describeWithFlags('matmul', ALL_ENVS, () => {
  it('A x B', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = dl.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = dl.matMul(a, b);

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(c, [0, 8, -3, 20]);
  });

  it('A x B^t', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = dl.tensor2d([1, 0, 2, 4, 3, 0], [2, 3]);

    const transposeA = false;
    const transposeB = true;
    const c = dl.matMul(a, b, transposeA, transposeB);

    const expected = [7, 10, 16, 31];
    expectArraysClose(c, expected);
  });

  it('A^t x B', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = dl.tensor2d([1, 0, 2, 4, 3, 0], [2, 3]);

    const transposeA = true;
    const transposeB = false;
    const c = dl.matMul(a, b, transposeA, transposeB);

    const expected = [17, 12, 2, 22, 15, 4, 27, 18, 6];
    expectArraysClose(c, expected);
  });

  it('A^t x B^t', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    const b = dl.tensor2d([1, 0, 2, 4, 3, 0], [2, 3]);

    const transposeA = true;
    const transposeB = true;
    const c = dl.matMul(a, b, transposeA, transposeB);

    const expected = [11, 13, 14, 20];
    expectArraysClose(c, expected);
  });

  it('A x B^t shapes do not match', () => {
    const a = dl.zeros<Rank.R2>([2, 3]);
    const b = dl.zeros<Rank.R2>([3, 2]);

    const f = () => {
      const transposeA = false;
      const transposeB = true;
      dl.matMul(a, b, transposeA, transposeB);
    };
    expect(f).toThrowError();
  });

  it('A^t x B shapes do not match', () => {
    const a = dl.zeros<Rank.R2>([2, 3]);
    const b = dl.zeros<Rank.R2>([3, 2]);

    const f = () => {
      const transposeA = true;
      const transposeB = false;
      dl.matMul(a, b, transposeA, transposeB);
    };
    expect(f).toThrowError();
  });

  it('A^t x B^t shapes do not match', () => {
    const a = dl.zeros<Rank.R2>([3, 2]);
    const b = dl.zeros<Rank.R2>([3, 2]);

    const f = () => {
      const transposeA = true;
      const transposeB = true;
      dl.matMul(a, b, transposeA, transposeB);
    };
    expect(f).toThrowError();
  });

  it('matmul throws when inner dimensions dont match', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = dl.tensor2d([0, 1, -3, 2, 2, 1, 2, 2], [4, 2]);

    expect(() => dl.matMul(a, b)).toThrowError();
  });

  it('matmul throws when passed non matrices', () => {
    // tslint:disable-next-line:no-any
    const a: any =
        dl.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const b = dl.tensor2d([0, 1, -3, 2, 2, 1, 2, 2], [4, 2]);

    expect(() => dl.matMul(a, b)).toThrowError();
    expect(() => dl.matMul(b, a)).toThrowError();
  });

  it('Vector times matrix', () => {
    const v = dl.tensor1d([2, 3]);
    const matrix = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const result = dl.vectorTimesMatrix(v, matrix);

    const expected = [11, 16];
    expectArraysClose(result, expected);
  });

  it('Vector times matrix with implicit reshape', () => {
    const v = dl.tensor1d([2, 3]);

    const matrix = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const result = dl.vectorTimesMatrix(v, matrix);

    const expected = [11, 16];
    expectArraysClose(result, expected);
  });

  it('Vector times matrix throws when not passed a vector', () => {
    // tslint:disable-next-line:no-any
    const v: any = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const matrix = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    expect(() => dl.vectorTimesMatrix(v, matrix)).toThrowError();
  });

  it('Vector times matrix throws when not passed a matrix', () => {
    const v = dl.tensor1d([2, 3]);
    // tslint:disable-next-line:no-any
    const matrix: any = dl.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);

    expect(() => dl.vectorTimesMatrix(v, matrix)).toThrowError();
  });

  it('Matrix times vector', () => {
    const matrix = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const v = dl.tensor1d([2, 3]);
    const result = dl.matrixTimesVector(matrix, v);

    const expected = [8, 18];
    expectArraysClose(result, expected);
  });

  it('Matrix * vector propagates NaNs', () => {
    const matrix = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const v = dl.tensor1d([2, NaN]);
    const result = dl.matrixTimesVector(matrix, v);

    const expected = [NaN, NaN];
    expectArraysClose(result, expected);
  });

  it('matrix times vector throws when not passed a vector', () => {
    // tslint:disable-next-line:no-any
    const v: any = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const matrix = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    expect(() => dl.matrixTimesVector(matrix, v)).toThrowError();
  });

  it('matrix times vector throws when not passed a matrix', () => {
    const v = dl.tensor1d([2, 3]);

    // tslint:disable-next-line:no-any
    const matrix: any = dl.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);

    expect(() => dl.matrixTimesVector(matrix, v)).toThrowError();
  });

  it('Dot product', () => {
    const v1 = dl.tensor1d([2, 3]);
    const v2 = dl.tensor1d([2, 1]);
    const result = MatmulOps.dotProduct(v1, v2);

    expectNumbersClose(result.get(), 7);
  });

  it('Dot product propagates NaNs', () => {
    const v1 = dl.tensor1d([2, NaN]);
    const v2 = dl.tensor1d([2, 1]);
    const result = MatmulOps.dotProduct(v1, v2);
    expect(result.get()).toEqual(NaN);
  });

  it('Dot product throws when vectors are different size', () => {
    const v1 = dl.tensor1d([2, 3, 3]);
    const v2 = dl.tensor1d([2, 1]);

    expect(() => MatmulOps.dotProduct(v1, v2)).toThrowError();
    expect(() => MatmulOps.dotProduct(v2, v1)).toThrowError();
  });

  it('Dot product throws when passed non vectors', () => {
    // tslint:disable-next-line:no-any
    const v1: any = dl.tensor2d([1, 2, 3, 3], [2, 2]);
    const v2 = dl.tensor1d([2, 1]);

    expect(() => MatmulOps.dotProduct(v1, v2)).toThrowError();
    expect(() => MatmulOps.dotProduct(v2, v1)).toThrowError();
  });

  it('Outer product', () => {
    const v1 = dl.tensor1d([2, 3]);
    const v2 = dl.tensor1d([2, 1]);
    const result = dl.outerProduct(v1, v2);

    const expected = [4, 2, 6, 3];
    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(result, expected);
  });

  it('gradients: A * B', () => {
    const a = dl.tensor2d([1, 2, 3, 10, 20, 30], [2, 3]);
    const b = dl.tensor2d([2, 3, 4, 1, 2, 3], [3, 2]);
    const dy = dl.tensor2d([1, 10, 20, 30], [2, 2]);

    const transposeA = false;
    const transposeB = false;
    const grads = dl.grads(

        (a: dl.Tensor2D, b: dl.Tensor2D) =>
            dl.matMul(a, b, transposeA, transposeB));
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
});

describeWithFlags('matmul webgl-only', WEBGL_ENVS, () => {
  it('Matrix times vector, large matrix', () => {
    const maxTexSize = 16000;
    const sharedDim = maxTexSize + 4;
    const matrix = dl.buffer<Rank.R2>([2, sharedDim], 'float32');
    matrix.set(1, 0, sharedDim - 3);
    matrix.set(1, 0, sharedDim - 2);

    const v = dl.buffer<Rank.R1>([sharedDim], 'float32');
    v.set(1, sharedDim - 3);
    v.set(1, sharedDim - 2);

    const result = dl.matrixTimesVector(matrix.toTensor(), v.toTensor());
    const expected = [2, 0];
    expectArraysClose(result, expected);
  });
});
