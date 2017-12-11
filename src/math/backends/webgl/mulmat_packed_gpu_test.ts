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

import * as test_util from '../../../test_util';
import {MatrixOrientation} from '../types/matmul';

import {GPGPUContext} from './gpgpu_context';
import * as mulmat_packed_gpu from './mulmat_packed_gpu';

describe('mulmat_packed_gpu (1x1 * 1x1)', () => {
  it('returns a 1x1 matrix', () => {
    const a = new Float32Array([0]);
    const b = new Float32Array([0]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, 1], b, [1, 1]);
    expect(result.length).toEqual(1);
  });

  it('returns [0] when multiplying [0] by [0]', () => {
    const a = new Float32Array([0]);
    const b = new Float32Array([0]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, 1], b, [1, 1]);
    expect(result[0]).toEqual(0);
  });

  it('returns [1] when multiplying [1] by [1]', () => {
    const a = new Float32Array([1]);
    const b = new Float32Array([1]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, 1], b, [1, 1]);
    expect(result[0]).toEqual(1);
  });

  it('returns [-1] when multiplying [1] by [-1]', () => {
    const a = new Float32Array([1]);
    const b = new Float32Array([-1]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, 1], b, [1, 1]);
    expect(result[0]).toEqual(-1);
  });

  it('returns [4.08] when multiplying [1.2] by [3.4]', () => {
    const a = new Float32Array([1.2]);
    const b = new Float32Array([3.4]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, 1], b, [1, 1]);
    test_util.expectNumbersClose(result[0], 4.08);
  });

  it('returns [356000] when multiplying [356] by [1000]', () => {
    const a = new Float32Array([356]);
    const b = new Float32Array([1000]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, 1], b, [1, 1]);
    expect(result[0]).toEqual(356000);
  });

  it('returns [-31415926] when multiplying [-3.1415926] by [10000000]', () => {
    const a = new Float32Array([-3.1415926]);
    const b = new Float32Array([10000000]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, 1], b, [1, 1]);
    expect(result[0]).toEqual(-31415926);
  });
});

describe('mulmat_packed_gpu (dot product)', () => {
  it('returns a 1x1 matrix', () => {
    const a = new Float32Array(5);
    const b = new Float32Array(5);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, a.length], b, [b.length, 1]);
    expect(result.length).toEqual(1);
  });

  it('returns zero when one vector is all zeroes', () => {
    const a = new Float32Array(5);
    const b = new Float32Array([1, 2, 3, 4, 5]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, a.length], b, [b.length, 1]);
    expect(result[0]).toEqual(0);
  });

  it('returns the sum of b when a is all ones', () => {
    const a = new Float32Array([1, 1, 1, 1, 1]);
    const b = new Float32Array([0, 1, 2, 3, 100]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, a.length], b, [b.length, 1]);
    expect(result[0]).toEqual(106);
  });

  it('computes the dot product of a and b', () => {
    const a = new Float32Array([10, 20, 30, 40, 50]);
    const b = new Float32Array([0.5, 1.1, 12.4, 32.5, -123.98]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, a.length], b, [b.length, 1]);
    const expected = test_util.cpuDotProduct(a, b);
    test_util.expectNumbersClose(result[0], expected);
  });

  it('computes a dot product on very large vectors', () => {
    const a: Float32Array = test_util.randomArrayInRange(2048, -1, 1);
    const b: Float32Array = test_util.randomArrayInRange(2048, -1, 1);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, a.length], b, [b.length, 1]);
    const expected = test_util.cpuDotProduct(a, b);
    test_util.expectNumbersClose(result[0], expected);
  });
});

function cpuMul2x2(a: Float32Array, b: Float32Array): Float32Array {
  if (a.length !== 4 || b.length !== 4) {
    throw new Error('a and b must have 4 elements.');
  }
  /*
   a = [0 1   b = [0 1
        2 3]       2 3]
   a[0] = [a0 a1] dot [b0 b2]
   a[1] = [a0 a1] dot [b1 b3]
   a[2] = [a2 a3] dot [b0 b2]
   a[3] = [a2 a3] dot [b1 b3]
   */
  const result = new Float32Array(4);
  result[0] = (a[0] * b[0]) + (a[1] * b[2]);
  result[1] = (a[0] * b[1]) + (a[1] * b[3]);
  result[2] = (a[2] * b[0]) + (a[3] * b[2]);
  result[3] = (a[2] * b[1]) + (a[3] * b[3]);
  return result;
}

describe('mulmat_packed_gpu (2x2 * 2x2)', () => {
  it('returns a 2x2 matrix', () => {
    const a = new Float32Array([0, 0, 0, 0]);
    const b = new Float32Array([0, 0, 0, 0]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 2], b, [2, 2]);
    expect(result.length).toEqual(4);
  });

  it('returns the identity when multiplying two identity matrices', () => {
    const a = test_util.makeIdentity(2);
    const b = test_util.makeIdentity(2);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 2], b, [2, 2]);
    expect(result).toEqual(cpuMul2x2(a, b));
  });

  it('returns [0] when A is [0]', () => {
    const a = new Float32Array([0, 0, 0, 0]);
    const b = new Float32Array([1, 2, 3, 4]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 2], b, [2, 2]);
    expect(result).toEqual(a);
  });

  it('returns [0] when B is [0]', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array(4);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 2], b, [2, 2]);
    expect(result).toEqual(b);
  });

  it('returns B when A is identity', () => {
    const a = test_util.makeIdentity(2);
    const b = new Float32Array([11, -22, 33.333, -44.44444]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 2], b, [2, 2]);
    expect(result).toEqual(b);
  });

  it('returns A when B is identity', () => {
    const a = new Float32Array([11, -22, 33.333, -44.44444]);
    const b = test_util.makeIdentity(2);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 2], b, [2, 2]);
    expect(result).toEqual(a);
  });

  it('returns the product of A and B when non-identity', () => {
    const a = new Float32Array([10000.02, -1.2, 3.14159, -2345.1234]);
    const b = new Float32Array([-23.45, 0.01234, 100, 2.5]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 2], b, [2, 2]);
    expect(result).toEqual(cpuMul2x2(a, b));
  });
});

describe('mulmat_packed_gpu (different shapes)', () => {
  it('returns a 4x1 when multiplying a 4x4 with a 4x1', () => {
    const a = new Float32Array(16);
    const b = new Float32Array(4);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [4, 4], b, [4, 1]);
    expect(result.length).toEqual(4);
  });

  it('returns B (4x1) when A (4x4) is I', () => {
    const a = test_util.makeIdentity(4);
    const b = new Float32Array([1, 2, 3, 4]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [4, 4], b, [4, 1]);
    expect(result).toEqual(b);
  });

  it('4x2 * 2x2', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const b = new Float32Array([9, 10, 11, 12]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [4, 2], b, [2, 2]);
    const expected = test_util.cpuMultiplyMatrix(a, 4, 2, b, 2, 2);
    test_util.expectArraysClose(result, expected);
  });

  it('multiplies a 4x1 by a non-identity 4x4', () => {
    const a = new Float32Array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const b = new Float32Array([1, 2, 3, 4]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [4, 4], b, [4, 1]);
    expect(result).toEqual(test_util.cpuMultiplyMatrix(a, 4, 4, b, 4, 1));
  });

  it('returns a 2x3 when multiplying a 2x4 by a 4x3', () => {
    const a = new Float32Array(8);
    const b = new Float32Array(12);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 4], b, [4, 3]);
    expect(result.length).toEqual(6);
  });

  it('multiplies A (2x4) by B(4x3)', () => {
    const a = new Float32Array([0.1, 3.2, -4.5, 11.78, -0.234, -2.999, 7, 9]);
    const b = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 4], b, [4, 3]);
    const expected = test_util.cpuMultiplyMatrix(a, 2, 4, b, 4, 3);
    test_util.expectArraysClose(result, expected);
  });
});

describe('mulmat_packed_gpu (large matrices)', () => {
  it('returns 128x128 when multiplying 2 128x128s', () => {
    const a = new Float32Array(128 * 128);
    const b = new Float32Array(128 * 128);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [128, 128], b, [128, 128]);
    expect(result.length).toEqual(128 * 128);
  });

  it('multiplies 2 128x128s', () => {
    const a = test_util.randomArrayInRange(128 * 128, -1, 1);
    const b = test_util.randomArrayInRange(128 * 128, -1, 1);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [128, 128], b, [128, 128]);
    const expected = test_util.cpuMultiplyMatrix(a, 128, 128, b, 128, 128);
    test_util.expectArraysClose(result, expected);
  });
});

describe('mulmat_packed_gpu (multiple matrices)', () => {
  it('4x2 * 2x12 * 12x1 === 4x1', () => {
    const aData = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    const bData = new Float32Array([
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    ]);
    const cData = new Float32Array([
      -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2
    ]);

    const gpgpu = new GPGPUContext();

    const axbProgram =
        gpgpu.createProgram(mulmat_packed_gpu.getFragmentShaderSource(
            2, MatrixOrientation.REGULAR, MatrixOrientation.REGULAR));
    const abxcProgram =
        gpgpu.createProgram(mulmat_packed_gpu.getFragmentShaderSource(
            12, MatrixOrientation.REGULAR, MatrixOrientation.REGULAR));

    const a: WebGLTexture = gpgpu.createPackedMatrixTexture(4, 2);
    const b: WebGLTexture = gpgpu.createPackedMatrixTexture(2, 12);
    const ab: WebGLTexture = gpgpu.createPackedMatrixTexture(4, 12);
    const c: WebGLTexture = gpgpu.createPackedMatrixTexture(12, 1);
    const r: WebGLTexture = gpgpu.createPackedMatrixTexture(4, 1);

    gpgpu.uploadMatrixToPackedTexture(a, 4, 2, aData);
    gpgpu.uploadMatrixToPackedTexture(b, 2, 12, bData);
    gpgpu.uploadMatrixToPackedTexture(c, 12, 1, cData);

    mulmat_packed_gpu.multiplyMatrixPacked(
        gpgpu, axbProgram, a, b, ab, [4, 12]);
    mulmat_packed_gpu.multiplyMatrixPacked(
        gpgpu, abxcProgram, ab, c, r, [4, 1]);

    const result = gpgpu.downloadMatrixFromPackedTexture(r, 4, 1);
    const expected = test_util.cpuMultiplyMatrix(
        test_util.cpuMultiplyMatrix(aData, 4, 2, bData, 2, 12), 4, 12, cData,
        12, 1);
    test_util.expectArraysClose(result, expected);

    gpgpu.deleteMatrixTexture(a);
    gpgpu.deleteMatrixTexture(b);
    gpgpu.deleteMatrixTexture(ab);
    gpgpu.deleteMatrixTexture(c);
    gpgpu.deleteMatrixTexture(r);
    gpgpu.deleteProgram(axbProgram);
    gpgpu.deleteProgram(abxcProgram);
    gpgpu.dispose();
  });
});

describe('mulmat_packed_gpu A * B^t', () => {
  it('1x1 * 1x1', () => {
    const a = new Float32Array([2]);
    const b = new Float32Array([3]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [1, 1], b, [1, 1], MatrixOrientation.REGULAR,
        MatrixOrientation.TRANSPOSED);
    expect(result[0]).toEqual(6);
  });

  it('2x2 * 2x2', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([5, 6, 7, 8]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 2], b, [2, 2], MatrixOrientation.REGULAR,
        MatrixOrientation.TRANSPOSED);

    const bt = new Float32Array([b[0], b[2], b[1], b[3]]);
    const expected = test_util.cpuMultiplyMatrix(a, 2, 2, bt, 2, 2);
    test_util.expectArraysClose(result, expected);
  });

  it('2x4 * 4x2', () => {
    /*
      A = [1 2 3 4   B = [ 9 10 11 12  B^t = [ 9 13
           5 6 7 8]       13 14 15 16]        10 14
                                              11 15
                                              12 16]

      A * B^t = [1*9+2*10+3*11+4*12 1*13+2*14+3*15+4*16
                 5*9+6*10+7*11+8*12 5*13+6*14+7*15+8*16]

              = [110 150
                 278 382
     */
    const a = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const b = new Float32Array([9, 10, 11, 12, 13, 14, 15, 16]);
    const result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 4], b, [2, 4], MatrixOrientation.REGULAR,
        MatrixOrientation.TRANSPOSED);

    const bt =
        new Float32Array([b[0], b[4], b[1], b[5], b[2], b[6], b[3], b[7]]);
    const expected = test_util.cpuMultiplyMatrix(a, 2, 4, bt, 4, 2);
    test_util.expectArraysClose(result, expected);
  });
});

describe('mulmat_packed_gpu (transposed versions)', () => {
  it('A * B^t', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([1, 0, 2, 4, 3, 0]);
    const c = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 3], b, [2, 3], MatrixOrientation.REGULAR,
        MatrixOrientation.TRANSPOSED);
    const expected = new Float32Array([7, 10, 16, 31]);
    expect(c).toEqual(expected);
  });

  it('A^t * B', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([1, 0, 2, 4, 3, 0]);
    const c = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [2, 3], b, [2, 3], MatrixOrientation.TRANSPOSED,
        MatrixOrientation.REGULAR);
    const expected = new Float32Array([17, 12, 2, 22, 15, 4, 27, 18, 6]);
    expect(c).toEqual(expected);
  });

  it('A^t * B^t', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([1, 0, 2, 4, 3, 0]);
    const c = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(
        a, [3, 2], b, [2, 3], MatrixOrientation.TRANSPOSED,
        MatrixOrientation.TRANSPOSED);
    const expected = new Float32Array([11, 13, 14, 20]);
    expect(c).toEqual(expected);
  });
});
