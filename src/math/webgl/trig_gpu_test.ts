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
import * as util from '../../util';
import {Array1D, Array2D, Array3D, Scalar} from '../ndarray';

import * as unaryop_gpu from './unaryop_gpu';
import * as unaryop_gpu_test from './unaryop_gpu_test';

describe('sin_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array2D.zeros([28, 28]);
    const result = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.SIN);
    expect(result.length).toEqual(a.size);
  });

  it('Sin equals CPU', () => {
    const size = 10;
    const a = test_util.randomArrayInRange(size, -1, 1);
    const expectedResult = new Float32Array(size);
    for (let i = 0; i < a.length; i++) {
      expectedResult[i] = Math.sin(a[i]);
    }
    const aArr = Array1D.new(a);
    const result = unaryop_gpu_test.uploadUnaryDownload(aArr, unaryop_gpu.SIN);
    test_util.expectArraysClose(result, expectedResult, 1e-3);
  });
});

describe('cos_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array2D.zeros([28, 28]);
    const result = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.COS);
    expect(result.length).toEqual(a.size);
  });

  it('Cos equals CPU', () => {
    const size = 10;
    const a = test_util.randomArrayInRange(size, -1, 1);
    const expectedResult = new Float32Array(size);
    for (let i = 0; i < a.length; i++) {
      expectedResult[i] = Math.cos(a[i]);
    }
    const aArr = Array1D.new(a);
    const result = unaryop_gpu_test.uploadUnaryDownload(aArr, unaryop_gpu.COS);
    test_util.expectArraysClose(result, expectedResult, 1e-3);
  });
});

describe('tan_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array2D.zeros([28, 28]);
    const result = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.TAN);
    expect(result.length).toEqual(a.size);
  });

  it('Tan equals CPU', () => {
    const size = 10;
    const a = test_util.randomArrayInRange(size, -1, 1);
    const expectedResult = new Float32Array(size);
    for (let i = 0; i < a.length; i++) {
      expectedResult[i] = Math.tan(a[i]);
    }
    const aArr = Array1D.new(a);
    const result = unaryop_gpu_test.uploadUnaryDownload(aArr, unaryop_gpu.TAN);
    test_util.expectArraysClose(result, expectedResult, 1e-3);
  });
});

describe('asin_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array2D.zeros([28, 28]);
    const result = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.ASIN);
    expect(result.length).toEqual(a.size);
  });

  it('asin equals CPU', () => {
    const size = 10;
    const a = test_util.randomArrayInRange(size, -1, 1);
    const expectedResult = new Float32Array(size);
    for (let i = 0; i < a.length; i++) {
      expectedResult[i] = Math.asin(a[i]);
    }
    const aArr = Array1D.new(a);
    const result = unaryop_gpu_test.uploadUnaryDownload(aArr, unaryop_gpu.ASIN);
    test_util.expectArraysClose(result, expectedResult, 1e-3);
  });
});

describe('acos_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array2D.zeros([28, 28]);
    const result = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.ACOS);
    expect(result.length).toEqual(a.size);
  });

  it('acos equals CPU', () => {
    const size = 10;
    const a = test_util.randomArrayInRange(size, -1, 1);
    const expectedResult = new Float32Array(size);
    for (let i = 0; i < a.length; i++) {
      expectedResult[i] = Math.acos(a[i]);
    }
    const aArr = Array1D.new(a);
    const result = unaryop_gpu_test.uploadUnaryDownload(aArr, unaryop_gpu.ACOS);
    test_util.expectArraysClose(result, expectedResult, 1e-3);
  });
});

describe('atan_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array2D.zeros([28, 28]);
    const result = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.ATAN);
    expect(result.length).toEqual(a.size);
  });

  it('atan equals CPU', () => {
    const size = 10;
    const a = test_util.randomArrayInRange(size, -1, 1);
    const expectedResult = new Float32Array(size);
    for (let i = 0; i < a.length; i++) {
      expectedResult[i] = Math.atan(a[i]);
    }
    const aArr = Array1D.new(a);
    const result = unaryop_gpu_test.uploadUnaryDownload(aArr, unaryop_gpu.ATAN);
    test_util.expectArraysClose(result, expectedResult, 1e-3);
  });
});

describe('sinh_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array2D.zeros([28, 28]);
    const result = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.SINH);
    expect(result.length).toEqual(a.size);
  });

  it('sinh equals CPU', () => {
    const size = 10;
    const a = test_util.randomArrayInRange(size, -1, 1);
    const expectedResult = new Float32Array(size);
    for (let i = 0; i < a.length; i++) {
      expectedResult[i] = Math.sinh(a[i]);
    }
    const aArr = Array1D.new(a);
    const result = unaryop_gpu_test.uploadUnaryDownload(aArr, unaryop_gpu.SINH);
    test_util.expectArraysClose(result, expectedResult, 1e-3);
  });

  it('sinh(0) = 0', () => {
    const a = Scalar.new(0);
    const r = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.SINH);
    expect(r).toBeCloseTo(0);
  });
});

describe('cosh_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array2D.zeros([28, 28]);
    const result = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.COSH);
    expect(result.length).toEqual(a.size);
  });

  it('cosh equals CPU', () => {
    const size = 10;
    const a = test_util.randomArrayInRange(size, -1, 1);
    const expectedResult = new Float32Array(size);
    for (let i = 0; i < a.length; i++) {
      expectedResult[i] = Math.cosh(a[i]);
    }
    const aArr = Array1D.new(a);
    const result = unaryop_gpu_test.uploadUnaryDownload(aArr, unaryop_gpu.COSH);
    test_util.expectArraysClose(result, expectedResult, 1e-3);
  });

  it('cosh(0) = 1', () => {
    const a = Scalar.new(0);
    const r = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.COSH);
    expect(r).toBeCloseTo(1);
  });
});

describe('tanh_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array3D.zeros([28, 14, 2]);
    const result = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.TANH);
    expect(result.length).toEqual(a.size);
  });

  it('Tanh equals CPU', () => {
    const size = 10;
    const a = test_util.randomArrayInRange(size, -1, 1);
    const expectedResult = new Float32Array(size);
    for (let i = 0; i < a.length; i++) {
      expectedResult[i] = util.tanh(a[i]);
    }
    const aArr = Array2D.new([2, 5], a);
    const result = unaryop_gpu_test.uploadUnaryDownload(aArr, unaryop_gpu.TANH);
    test_util.expectArraysClose(result, expectedResult);
  });

  it('overflow', () => {
    const a = Scalar.new(100);
    const r = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.TANH);
    expect(r).toBeCloseTo(1);
  });

  it('tanh(0) = 0', () => {
    const a = Scalar.new(0);
    const r = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.TANH);
    expect(r).toBeCloseTo(0);
  });

  it('tanh(0.01) is close to 0.01', () => {
    const a = Scalar.new(0.01);
    const r = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.TANH);
    expect(r).toBeCloseTo(0.01);
  });

  it('underflow', () => {
    const a = Scalar.new(-100);
    const r = unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.TANH);
    expect(r).toBeCloseTo(-1);
  });
});
