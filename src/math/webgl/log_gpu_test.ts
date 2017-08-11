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
import {UnaryOp} from './unaryop_gpu';
import * as unaryop_gpu_test from './unaryop_gpu_test';
import {Array2D} from '../ndarray';

describe('log_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = new Float32Array(23 * 32);
    const result = uploadLogDownload(a, 23, 32);
    expect(result.length).toEqual(a.length);
  });

  it('returns 1.0 when the only value in a 1x1 matrix is e', () => {
    const a = new Float32Array([Math.E]);
    const result = uploadLogDownload(a, 1, 1);
    expect(result[0]).toBeCloseTo(1.0);
  });

  it('operates on every value in a matrix', () => {
    const a = new Float32Array(6);
    a.fill(Math.E);
    const result = uploadLogDownload(a, 1, a.length);
    const expected = new Float32Array(a.length);
    expected.fill(1.0);
    test_util.expectArraysClose(result, expected, 0.0001);
  });

  it('calculates f(x)=ln x for every value in the matrix', () => {
    const a = new Float32Array([0.5, 1, 2, 3]);
    const result = uploadLogDownload(a, 1, a.length);
    const expected = new Float32Array(a.length);
    for (let i = 0; i < a.length; ++i) {
      expected[i] = Math.log(a[i]);
    }
    test_util.expectArraysClose(result, expected, 0.0001);
  });
});

function uploadLogDownload(
    a: Float32Array, rows: number, cols: number): Float32Array {
  const arr = Array2D.new([rows, cols], a);
  return unaryop_gpu_test.uploadUnaryDownload(arr, UnaryOp.LOG);
}
