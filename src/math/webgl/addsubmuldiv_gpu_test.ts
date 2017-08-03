/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as test_util from '../../test_util';
import {MatrixOrientation} from '../math';

import * as addsubmuldiv_gpu from './addsubmuldiv_gpu';

describe('addsubmuldiv_gpu ScalarPlusMatrix', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = new Float32Array(12 * 513);
    const result =
        addsubmuldiv_gpu.uploadScalarPlusMatrixDownload(0, a, [12, 513]);
    expect(result.length).toEqual(a.length);
  });

  it('preserves the matrix when the scalar is 0', () => {
    const a = new Float32Array([1, 2, 3]);
    const result =
        addsubmuldiv_gpu.uploadScalarPlusMatrixDownload(0, a, [1, 3]);
    test_util.expectArraysClose(result, a, 0);
  });

  it('adds the scalar to every element in the matrix', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const result =
        addsubmuldiv_gpu.uploadScalarPlusMatrixDownload(0.5, a, [2, 2]);
    test_util.expectArraysClose(
        result, new Float32Array([1.5, 2.5, 3.5, 4.5]), 0.0001);
  });
});

describe('addsubmuldiv_gpu MatrixMinusScalar', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = new Float32Array(12 * 513);
    const result =
        addsubmuldiv_gpu.uploadMatrixMinusScalarDownload(a, [12, 513], 0);
    expect(result.length).toEqual(a.length);
  });

  it('preserves the matrix when the scalar is 0', () => {
    const a = new Float32Array([1, 2, 3]);
    const result =
        addsubmuldiv_gpu.uploadMatrixMinusScalarDownload(a, [1, 3], 0);
    test_util.expectArraysClose(result, a, 0);
  });

  it('subtracts the scalar from every element in the matrix', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const result =
        addsubmuldiv_gpu.uploadMatrixMinusScalarDownload(a, [2, 2], 0.5);
    test_util.expectArraysClose(
        result, new Float32Array([0.5, 1.5, 2.5, 3.5]), 0.0001);
  });
});

describe('addsubmuldiv_gpu ScalarMinusMatrix', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = new Float32Array(12 * 513);
    const result =
        addsubmuldiv_gpu.uploadScalarMinusMatrixDownload(0, a, [12, 513]);
    expect(result.length).toEqual(a.length);
  });

  it('negates the matrix when the scalar is 0', () => {
    const a = new Float32Array([1, 2, 3]);
    const result =
        addsubmuldiv_gpu.uploadScalarMinusMatrixDownload(0, a, [1, 3]);
    test_util.expectArraysClose(result, new Float32Array([-1, -2, -3]), 0);
  });

  it('subtracts the matrix value from the scalar for every element', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const result =
        addsubmuldiv_gpu.uploadScalarMinusMatrixDownload(0.5, a, [2, 2]);
    test_util.expectArraysClose(
        result, new Float32Array([-0.5, -1.5, -2.5, -3.5]), 0.0001);
  });
});

describe('addsubmuldiv_gpu ScalarTimesMatrix', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = new Float32Array(12 * 513);
    const result =
        addsubmuldiv_gpu.uploadScalarTimesMatrixDownload(0, a, [12, 513]);
    expect(result.length).toEqual(a.length);
  });

  it('zeros out the matrix when the scalar is 0', () => {
    const a = new Float32Array([1, 2, 3]);
    const result =
        addsubmuldiv_gpu.uploadScalarTimesMatrixDownload(0, a, [1, 3]);
    test_util.expectArraysClose(result, new Float32Array([0, 0, 0]), 0);
  });

  it('triples the matrix when the scalar is 3', () => {
    const a = new Float32Array([1, 2, 3]);
    const result =
        addsubmuldiv_gpu.uploadScalarTimesMatrixDownload(3, a, [1, 3]);
    test_util.expectArraysClose(result, new Float32Array([3, 6, 9]), 0);
  });
});

describe('addsubmuldiv_gpu element-wise matrix product', () => {
  function cpuMultiply(a: Float32Array, b: Float32Array): Float32Array {
    const result = new Float32Array(a.length);
    for (let i = 0; i < result.length; ++i) {
      result[i] = a[i] * b[i];
    }
    return result;
  }

  it('returns a matrix the size of A (and B)', () => {
    const a = new Float32Array(1234);
    const b = new Float32Array(1234);
    const result =
        addsubmuldiv_gpu.uploadMatrixTimesMatrixDownload(a, b, [1234 / 2, 2]);
    expect(result.length).toEqual(a.length);
  });

  it('sets all result entries to 0 if A is 0', () => {
    const a = new Float32Array(257 * 257);
    const b = new Float32Array(a.length);
    b.fill(1.0);
    const result =
        addsubmuldiv_gpu.uploadMatrixTimesMatrixDownload(a, b, [257, 257]);
    expect(result).toEqual(a);
  });

  it('sets all result entries to 0 if B is 0', () => {
    const a = new Float32Array(257 * 257);
    const b = new Float32Array(a.length);
    a.fill(1.0);
    const result =
        addsubmuldiv_gpu.uploadMatrixTimesMatrixDownload(a, b, [257, 257]);
    expect(result).toEqual(b);
  });

  it('sets all result entries to A if B is [1]', () => {
    const a = test_util.randomArrayInRange(16, -10, 10);
    const b = new Float32Array(16);
    b.fill(1.0);
    const result =
        addsubmuldiv_gpu.uploadMatrixTimesMatrixDownload(a, b, [4, 4]);
    test_util.expectArraysClose(a, result, 0.0001);
  });

  it('sets all result entries to B if A is [1]', () => {
    const a = new Float32Array(16);
    a.fill(1.0);
    const b = test_util.randomArrayInRange(16, -10, 10);
    const result =
        addsubmuldiv_gpu.uploadMatrixTimesMatrixDownload(a, b, [4, 4]);
    test_util.expectArraysClose(b, result, 0.0001);
  });

  it('writes the element-wise product of A and B to result', () => {
    const a = test_util.randomArrayInRange(64, -10, 10);
    const b = test_util.randomArrayInRange(64, -10, 10);
    const result =
        addsubmuldiv_gpu.uploadMatrixTimesMatrixDownload(a, b, [8, 8]);
    const expected = cpuMultiply(a, b);
    test_util.expectArraysClose(result, expected, 0.0001);
  });

  it('writes the element-wise product A * B^T to result', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([3, 1, 0, 2]);

    const result = addsubmuldiv_gpu.uploadMatrixTimesMatrixDownload(
        a, b, [2, 2], MatrixOrientation.REGULAR, MatrixOrientation.TRANSPOSED);

    const bTransposed = new Float32Array([3, 0, 1, 2]);
    const expected = cpuMultiply(a, bTransposed);
    test_util.expectArraysClose(result, expected, 0.0001);
  });
});

describe('addsubmuldiv_gpu element-wise matrix addition', () => {
  it('writes the element-wise A + B^T to result', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([3, 1, 0, 2]);

    const result = addsubmuldiv_gpu.uploadMatrixPlusMatrixDownload(
        a, b, [2, 2], MatrixOrientation.REGULAR, MatrixOrientation.TRANSPOSED);

    const expected = new Float32Array([4, 2, 4, 6]);
    test_util.expectArraysClose(result, expected, 0.0001);
  });

  it('writes the element-wise A^T + B^T to result', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([3, 1, 0, 2]);

    const result = addsubmuldiv_gpu.uploadMatrixPlusMatrixDownload(
        a, b, [2, 2], MatrixOrientation.TRANSPOSED,
        MatrixOrientation.TRANSPOSED);

    const expected = new Float32Array([4, 3, 3, 6]);
    test_util.expectArraysClose(result, expected, 0.0001);
  });
});
