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

import * as test_util from '../../test_util';
import {MatrixOrientation} from '../math';
import {Array2D, initializeGPU} from '../ndarray';

import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {MatMulProgram} from './mulmat_gpu';
import {TextureManager} from './texture_manager';

describe('mulmat_gpu (1x1 * 1x1)', () => {
  it('returns a 1x1 matrix', () => {
    const a = new Float32Array([0]);
    const b = new Float32Array([0]);
    const result = uploadMultiplyMatrixDownload(a, 1, 1, b, 1, 1);
    expect(result.length).toEqual(1);
  });

  it('returns [0] when multiplying [0] by [0]', () => {
    const a = new Float32Array([0]);
    const b = new Float32Array([0]);
    const result = uploadMultiplyMatrixDownload(a, 1, 1, b, 1, 1);
    expect(result[0]).toEqual(0);
  });

  it('returns [1] when multiplying [1] by [1]', () => {
    const a = new Float32Array([1]);
    const b = new Float32Array([1]);
    const result = uploadMultiplyMatrixDownload(a, 1, 1, b, 1, 1);
    expect(result[0]).toEqual(1);
  });

  it('returns [-1] when multiplying [1] by [-1]', () => {
    const a = new Float32Array([1]);
    const b = new Float32Array([-1]);
    const result = uploadMultiplyMatrixDownload(a, 1, 1, b, 1, 1);
    expect(result[0]).toEqual(-1);
  });

  it('returns [4.08] when multiplying [1.2] by [3.4]', () => {
    const a = new Float32Array([1.2]);
    const b = new Float32Array([3.4]);
    const result = uploadMultiplyMatrixDownload(a, 1, 1, b, 1, 1);
    expect(result[0]).toBeCloseTo(4.08);
  });

  it('returns [356000] when multiplying [356] by [1000]', () => {
    const a = new Float32Array([356]);
    const b = new Float32Array([1000]);
    const result = uploadMultiplyMatrixDownload(a, 1, 1, b, 1, 1);
    expect(result[0]).toEqual(356000);
  });

  it('returns [-31415926] when multiplying [-3.1415926] by [10000000]', () => {
    const a = new Float32Array([-3.1415926]);
    const b = new Float32Array([10000000]);
    const result = uploadMultiplyMatrixDownload(a, 1, 1, b, 1, 1);
    expect(result[0]).toEqual(-31415926);
  });
});

describe('mulmat_gpu (dot product)', () => {
  it('returns a 1x1 matrix', () => {
    const a = new Float32Array(5);
    const b = new Float32Array(5);
    const result = uploadMultiplyMatrixDownload(a, 1, a.length, b, b.length, 1);
    expect(result.length).toEqual(1);
  });

  it('returns zero when one vector is all zeroes', () => {
    const a = new Float32Array(5);
    const b = new Float32Array([1, 2, 3, 4, 5]);
    const result = uploadMultiplyMatrixDownload(a, 1, a.length, b, b.length, 1);
    expect(result[0]).toEqual(0);
  });

  it('returns the sum of b when a is all ones', () => {
    const a = new Float32Array([1, 1, 1, 1, 1]);
    const b = new Float32Array([0, 1, 2, 3, 100]);
    const result = uploadMultiplyMatrixDownload(a, 1, a.length, b, b.length, 1);
    expect(result[0]).toEqual(106);
  });

  it('computes the dot product of a and b', () => {
    const a = new Float32Array([10, 20, 30, 40, 50]);
    const b = new Float32Array([0.5, 1.1, 12.4, 32.5, -123.98]);
    const result = uploadMultiplyMatrixDownload(a, 1, a.length, b, b.length, 1);
    const expected = test_util.cpuDotProduct(a, b);
    expect(result[0]).toBeCloseTo(expected);
  });

  it('computes a dot product on very large vectors', () => {
    const a: Float32Array = test_util.randomArrayInRange(2048, -1, 1);
    const b: Float32Array = test_util.randomArrayInRange(2048, -1, 1);
    const result = uploadMultiplyMatrixDownload(a, 1, a.length, b, b.length, 1);
    const expected = test_util.cpuDotProduct(a, b);
    expect(result[0]).toBeCloseTo(expected);
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

describe('mulmat_gpu (2x2 * 2x2)', () => {
  it('returns a 2x2 matrix', () => {
    const a = new Float32Array([0, 0, 0, 0]);
    const b = new Float32Array([0, 0, 0, 0]);
    const result = uploadMultiplyMatrixDownload(a, 2, 2, b, 2, 2);
    expect(result.length).toEqual(4);
  });

  it('returns the identity when multiplying two identity matrices', () => {
    const a = test_util.makeIdentity(2);
    const b = test_util.makeIdentity(2);
    const result = uploadMultiplyMatrixDownload(a, 2, 2, b, 2, 2);
    expect(result).toEqual(cpuMul2x2(a, b));
  });

  it('returns [0] when A is [0]', () => {
    const a = new Float32Array([0, 0, 0, 0]);
    const b = new Float32Array([1, 2, 3, 4]);
    const result = uploadMultiplyMatrixDownload(a, 2, 2, b, 2, 2);
    expect(result).toEqual(a);
  });

  it('returns [0] when B is [0]', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array(4);
    const result = uploadMultiplyMatrixDownload(a, 2, 2, b, 2, 2);
    expect(result).toEqual(b);
  });

  it('returns B when A is identity', () => {
    const a = test_util.makeIdentity(2);
    const b = new Float32Array([11, -22, 33.333, -44.44444]);
    const result = uploadMultiplyMatrixDownload(a, 2, 2, b, 2, 2);
    expect(result).toEqual(b);
  });

  it('returns A when B is identity', () => {
    const a = new Float32Array([11, -22, 33.333, -44.44444]);
    const b = test_util.makeIdentity(2);
    const result = uploadMultiplyMatrixDownload(a, 2, 2, b, 2, 2);
    expect(result).toEqual(a);
  });

  it('returns the product of A and B when non-identity', () => {
    const a = new Float32Array([10000.02, -1.2, 3.14159, -2345.1234]);
    const b = new Float32Array([-23.45, 0.01234, 100, 2.5]);
    const result = uploadMultiplyMatrixDownload(a, 2, 2, b, 2, 2);
    expect(result).toEqual(cpuMul2x2(a, b));
  });
});

describe('mulmat_gpu (different shapes)', () => {
  it('returns a 4x1 when multiplying a 4x4 with a 4x1', () => {
    const a = new Float32Array(16);
    const b = new Float32Array(4);
    const result = uploadMultiplyMatrixDownload(a, 4, 4, b, 4, 1);
    expect(result.length).toEqual(4);
  });

  it('returns B (4x1) when A (4x4) is I', () => {
    const a = test_util.makeIdentity(4);
    const b = new Float32Array([1, 2, 3, 4]);
    const result = uploadMultiplyMatrixDownload(a, 4, 4, b, 4, 1);
    expect(result).toEqual(b);
  });

  it('multiplies a 4x1 by a non-identity 4x4', () => {
    const a = new Float32Array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const b = new Float32Array([1, 2, 3, 4]);
    const result = uploadMultiplyMatrixDownload(a, 4, 4, b, 4, 1);
    expect(result).toEqual(test_util.cpuMultiplyMatrix(a, 4, 4, b, 4, 1));
  });

  it('returns a 2x3 when multiplying a 2x4 by a 4x3', () => {
    const a = new Float32Array(8);
    const b = new Float32Array(12);
    const result = uploadMultiplyMatrixDownload(a, 2, 4, b, 4, 3);
    expect(result.length).toEqual(6);
  });

  it('multiplies A (2x4) by B(4x3)', () => {
    const a = new Float32Array([0.1, 3.2, -4.5, 11.78, -0.234, -2.999, 7, 9]);
    const b = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const result = uploadMultiplyMatrixDownload(a, 2, 4, b, 4, 3);
    const expected = test_util.cpuMultiplyMatrix(a, 2, 4, b, 4, 3);
    test_util.expectArraysClose(result, expected);
  });
});

describe('mulmat_gpu (large matrices)', () => {
  it('returns 128x128 when multiplying 2 128x128s', () => {
    const a = new Float32Array(128 * 128);
    const b = new Float32Array(128 * 128);
    const result = uploadMultiplyMatrixDownload(a, 128, 128, b, 128, 128);
    expect(result.length).toEqual(128 * 128);
  });

  it('multiplies 2 128x128s', () => {
    const a = test_util.randomArrayInRange(128 * 128, -1, 1);
    const b = test_util.randomArrayInRange(128 * 128, -1, 1);
    const result = uploadMultiplyMatrixDownload(a, 128, 128, b, 128, 128);
    const expected = test_util.cpuMultiplyMatrix(a, 128, 128, b, 128, 128);
    test_util.expectArraysClose(result, expected);
  });
});

describe('mulmat_gpu (multiple matrices)', () => {
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

    const aShape: [number, number] = [4, 2];
    const bShape: [number, number] = [2, 12];
    const abShape: [number, number] = [aShape[0], bShape[1]];
    const cShape: [number, number] = [12, 1];
    const rShape: [number, number] = [aShape[0], cShape[1]];

    const a: WebGLTexture = gpgpu.createMatrixTexture(aShape[0], aShape[1]);
    const b: WebGLTexture = gpgpu.createMatrixTexture(bShape[0], bShape[1]);
    const ab: WebGLTexture = gpgpu.createMatrixTexture(abShape[0], abShape[1]);
    const c: WebGLTexture = gpgpu.createMatrixTexture(cShape[0], cShape[1]);
    const r: WebGLTexture = gpgpu.createMatrixTexture(rShape[0], rShape[1]);

    const aArr = new Array2D(aShape, {texture: a, textureShapeRC: aShape});
    const bArr = new Array2D(bShape, {texture: b, textureShapeRC: bShape});
    const abArr = new Array2D(abShape, {texture: ab, textureShapeRC: abShape});
    const cArr = new Array2D(cShape, {texture: c, textureShapeRC: cShape});
    const rArr = new Array2D(rShape, {texture: r, textureShapeRC: rShape});
    const matMulProgram = new MatMulProgram(aArr.shape, bArr.shape);
    const axbProgram =
        gpgpu_math.compileProgram(gpgpu, matMulProgram, [aArr, bArr], abArr);
    const matMulProgram2 = new MatMulProgram(abArr.shape, cArr.shape);
    const abxcProgram =
        gpgpu_math.compileProgram(gpgpu, matMulProgram2, [abArr, cArr], rArr);

    gpgpu.uploadMatrixToTexture(a, aShape[0], aShape[1], aData);
    gpgpu.uploadMatrixToTexture(b, bShape[0], bShape[1], bData);
    gpgpu.uploadMatrixToTexture(c, cShape[0], cShape[1], cData);

    gpgpu_math.runProgram(axbProgram, [aArr, bArr], abArr);
    gpgpu_math.runProgram(abxcProgram, [abArr, cArr], rArr);
    const result = gpgpu.downloadMatrixFromTexture(r, rShape[0], rShape[1]);
    const expected = test_util.cpuMultiplyMatrix(
        test_util.cpuMultiplyMatrix(aData, 4, 2, bData, 2, 12), 4, 12, cData,
        12, 1);
    test_util.expectArraysClose(result, expected);

    gpgpu.deleteMatrixTexture(a);
    gpgpu.deleteMatrixTexture(b);
    gpgpu.deleteMatrixTexture(ab);
    gpgpu.deleteMatrixTexture(c);
    gpgpu.deleteMatrixTexture(r);
    gpgpu.deleteProgram(axbProgram.webGLProgram);
    gpgpu.deleteProgram(abxcProgram.webGLProgram);
    gpgpu.dispose();
  });
});

describe('mulmat_gpu (transposed versions)', () => {
  it('A * B^t', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([1, 0, 2, 4, 3, 0]);
    const c = uploadMultiplyMatrixDownload(
        a, 2, 3, b, 2, 3, MatrixOrientation.REGULAR,
        MatrixOrientation.TRANSPOSED);
    const expected = new Float32Array([7, 10, 16, 31]);
    expect(c).toEqual(expected);
  });

  it('A^t * B', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([1, 0, 2, 4, 3, 0]);
    const c = uploadMultiplyMatrixDownload(
        a, 2, 3, b, 2, 3, MatrixOrientation.TRANSPOSED,
        MatrixOrientation.REGULAR);
    const expected = new Float32Array([17, 12, 2, 22, 15, 4, 27, 18, 6]);
    expect(c).toEqual(expected);
  });

  it('A^t * B^t', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([1, 0, 2, 4, 3, 0]);
    const c = uploadMultiplyMatrixDownload(
        a, 3, 2, b, 2, 3, MatrixOrientation.TRANSPOSED,
        MatrixOrientation.TRANSPOSED);
    const expected = new Float32Array([11, 13, 14, 20]);
    expect(c).toEqual(expected);
  });
});

describe('mulmat_gpu huge matrix', () => {
  it('vector times matrix', () => {
    const sharedDim = 1000;
    const outDim = 50000;
    const a = test_util.randomArrayInRange(sharedDim, -1, 1);
    const matrix = test_util.randomArrayInRange(sharedDim * outDim, -1, 1);
    const result = uploadMultiplyMatrixDownload(
        a, 1, sharedDim, matrix, sharedDim, outDim);
    const cpuResult =
        test_util.cpuMultiplyMatrix(a, 1, sharedDim, matrix, sharedDim, outDim);
    test_util.expectArraysClose(result, cpuResult);
  });
});

export function uploadMultiplyMatrixDownload(
    a: Float32Array, aNumRows: number, aNumCols: number, b: Float32Array,
    bNumRows: number, bNumCols: number,
    aOrientation = MatrixOrientation.REGULAR,
    bOrientation = MatrixOrientation.REGULAR): Float32Array {
  const gpgpu = new GPGPUContext();
  const texManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, texManager);

  const aShape: [number, number] = [aNumRows, aNumCols];
  const bShape: [number, number] = [bNumRows, bNumCols];

  const program = new MatMulProgram(aShape, bShape, aOrientation, bOrientation);
  const resArr = Array2D.zeros(program.outputShape as [number, number]);
  const aArr = Array2D.new(aShape, a);
  const bArr = Array2D.new(bShape, b);

  const binary =
      gpgpu_math.compileProgram(gpgpu, program, [aArr, bArr], resArr);
  gpgpu_math.runProgram(binary, [aArr, bArr], resArr);
  const result = resArr.getValues();

  aArr.dispose();
  bArr.dispose();
  texManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();

  return result;
}
