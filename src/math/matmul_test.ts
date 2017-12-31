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

import * as test_util from '../test_util';
import {MathTests} from '../test_util';
import {MatrixOrientation} from './backends/types/matmul';
import {Array1D, Array2D, Array3D} from './ndarray';

const commonTests: MathTests = it => {
  it('A x B', math => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = Array2D.new([3, 2], [0, 1, -3, 2, 2, 1]);

    const c = math.matMul(a, b);

    expect(c.shape).toEqual([2, 2]);
    test_util.expectArraysClose(c, [0, 8, -3, 20]);
  });

  it('A x B^t', math => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = Array2D.new([2, 3], [1, 0, 2, 4, 3, 0]);

    const c = math.matMul(
        a, b, MatrixOrientation.REGULAR, MatrixOrientation.TRANSPOSED);

    const expected = [7, 10, 16, 31];
    test_util.expectArraysClose(c, expected);
  });

  it('A^t x B', math => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = Array2D.new([2, 3], [1, 0, 2, 4, 3, 0]);

    const c = math.matMul(
        a, b, MatrixOrientation.TRANSPOSED, MatrixOrientation.REGULAR);

    const expected = [17, 12, 2, 22, 15, 4, 27, 18, 6];
    test_util.expectArraysClose(c, expected);
  });

  it('A^t x B^t', math => {
    const a = Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
    const b = Array2D.new([2, 3], [1, 0, 2, 4, 3, 0]);

    const c = math.matMul(
        a, b, MatrixOrientation.TRANSPOSED, MatrixOrientation.TRANSPOSED);

    const expected = [11, 13, 14, 20];
    test_util.expectArraysClose(c, expected);
  });

  it('A x B^t shapes do not match', math => {
    const a = Array2D.zeros([2, 3]);
    const b = Array2D.zeros([3, 2]);

    const f = () => {
      math.matMul(
          a, b, MatrixOrientation.REGULAR, MatrixOrientation.TRANSPOSED);
    };
    expect(f).toThrowError();
  });

  it('A^t x B shapes do not match', math => {
    const a = Array2D.zeros([2, 3]);
    const b = Array2D.zeros([3, 2]);

    const f = () => {
      math.matMul(
          a, b, MatrixOrientation.TRANSPOSED, MatrixOrientation.REGULAR);
    };
    expect(f).toThrowError();
  });

  it('A^t x B^t shapes do not match', math => {
    const a = Array2D.zeros([3, 2]);
    const b = Array2D.zeros([3, 2]);

    const f = () => {
      math.matMul(
          a, b, MatrixOrientation.TRANSPOSED, MatrixOrientation.TRANSPOSED);
    };
    expect(f).toThrowError();
  });

  it('matmul throws when inner dimensions dont match', math => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = Array2D.new([4, 2], [0, 1, -3, 2, 2, 1, 2, 2]);

    expect(() => math.matMul(a, b)).toThrowError();
  });

  it('matmul throws when passed non matrices', math => {
    // tslint:disable-next-line:no-any
    const a: any =
        Array3D.new([2, 3, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const b = Array2D.new([4, 2], [0, 1, -3, 2, 2, 1, 2, 2]);

    expect(() => math.matMul(a, b)).toThrowError();
    expect(() => math.matMul(b, a)).toThrowError();
  });

  it('Vector times matrix', math => {
    const v = Array1D.new([2, 3]);
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    const result = math.vectorTimesMatrix(v, matrix);

    const expected = [11, 16];
    test_util.expectArraysClose(result, expected);
  });

  it('Vector times matrix with implicit reshape', math => {
    const v = Array1D.new([2, 3]);

    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    const result = math.vectorTimesMatrix(v, matrix);

    const expected = [11, 16];
    test_util.expectArraysClose(result, expected);
  });

  it('Vector times matrix throws when not passed a vector', math => {
    // tslint:disable-next-line:no-any
    const v: any = Array2D.new([2, 2], [1, 2, 3, 4]);
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);

    expect(() => math.vectorTimesMatrix(v, matrix)).toThrowError();
  });

  it('Vector times matrix throws when not passed a matrix', math => {
    const v = Array1D.new([2, 3]);
    // tslint:disable-next-line:no-any
    const matrix: any = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);

    expect(() => math.vectorTimesMatrix(v, matrix)).toThrowError();
  });

  it('Matrix times vector', math => {
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    const v = Array1D.new([2, 3]);
    const result = math.matrixTimesVector(matrix, v);

    const expected = [8, 18];
    test_util.expectArraysClose(result, expected);
  });

  it('Matrix * vector propagates NaNs', math => {
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    const v = Array1D.new([2, NaN]);
    const result = math.matrixTimesVector(matrix, v);

    const expected = [NaN, NaN];
    test_util.expectArraysClose(result, expected);
  });

  it('matrix times vector throws when not passed a vector', math => {
    // tslint:disable-next-line:no-any
    const v: any = Array2D.new([2, 2], [1, 2, 3, 4]);
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);

    expect(() => math.matrixTimesVector(matrix, v)).toThrowError();
  });

  it('matrix times vector throws when not passed a matrix', math => {
    const v = Array1D.new([2, 3]);

    // tslint:disable-next-line:no-any
    const matrix: any = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);

    expect(() => math.matrixTimesVector(matrix, v)).toThrowError();
  });

  it('Dot product', math => {
    const v1 = Array1D.new([2, 3]);
    const v2 = Array1D.new([2, 1]);
    const result = math.dotProduct(v1, v2);

    test_util.expectNumbersClose(result.get(), 7);
  });

  it('Dot product propagates NaNs', math => {
    const v1 = Array1D.new([2, NaN]);
    const v2 = Array1D.new([2, 1]);
    const result = math.dotProduct(v1, v2);
    expect(result.get()).toEqual(NaN);
  });

  it('Dot product throws when vectors are different size', math => {
    const v1 = Array1D.new([2, 3, 3]);
    const v2 = Array1D.new([2, 1]);

    expect(() => math.dotProduct(v1, v2)).toThrowError();
    expect(() => math.dotProduct(v2, v1)).toThrowError();
  });

  it('Dot product throws when passed non vectors', math => {
    // tslint:disable-next-line:no-any
    const v1: any = Array2D.new([2, 2], [1, 2, 3, 3]);
    const v2 = Array1D.new([2, 1]);

    expect(() => math.dotProduct(v1, v2)).toThrowError();
    expect(() => math.dotProduct(v2, v1)).toThrowError();
  });

  it('Outer product', math => {
    const v1 = Array1D.new([2, 3]);
    const v2 = Array1D.new([2, 1]);
    const result = math.outerProduct(v1, v2);

    const expected = [4, 2, 6, 3];
    expect(result.shape).toEqual([2, 2]);
    test_util.expectArraysClose(result, expected);
  });

  // TODO(nsthorat): fix the precision for backprop.
  it('gradients: A * B', math => {
    const a = Array2D.new([2, 3], [1, 2, 3, 10, 20, 30]);
    const b = Array2D.new([3, 2], [2, 3, 4, 1, 2, 3]);
    const dy = Array2D.new([2, 2], [1, 10, 20, 30]);

    const gradients = math.vjp(
        () => math.matMul(
            a, b, MatrixOrientation.REGULAR, MatrixOrientation.REGULAR),
        {a, b}, dy);

    // da = dy * bT
    test_util.expectArraysClose(
        gradients.a,
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
    expect(gradients.b.shape).toEqual(b.shape);
    test_util.expectArraysClose(
        gradients.b,
        [
          a.get(0, 0) * dy.get(0, 0) + a.get(1, 0) * dy.get(1, 0),
          a.get(0, 0) * dy.get(0, 1) + a.get(1, 0) * dy.get(1, 1),
          a.get(0, 1) * dy.get(0, 0) + a.get(1, 1) * dy.get(1, 0),
          a.get(0, 1) * dy.get(0, 1) + a.get(1, 1) * dy.get(1, 1),
          a.get(0, 2) * dy.get(0, 0) + a.get(1, 2) * dy.get(1, 0),
          a.get(0, 2) * dy.get(0, 1) + a.get(1, 2) * dy.get(1, 1)
        ],
        1e-1);
  });
};

const gpuTests: MathTests = it => {
  it('Matrix times vector, large matrix', math => {
    const maxTexSize = 16000;
    const sharedDim = maxTexSize + 4;
    const matrix = Array2D.zeros([2, sharedDim]);
    matrix.set(1, 0, sharedDim - 3);
    matrix.set(1, 0, sharedDim - 2);

    const v = Array1D.zeros([sharedDim]);
    v.set(1, sharedDim - 3);
    v.set(1, sharedDim - 2);

    const result = math.matrixTimesVector(matrix, v);
    const expected = [2, 0];
    test_util.expectArraysClose(result, expected);
  });
};

test_util.describeMathCPU('matMul', [commonTests]);
test_util.describeMathGPU('matMul', [commonTests, gpuTests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
