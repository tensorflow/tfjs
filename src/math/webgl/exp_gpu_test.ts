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
import * as exp_gpu from './exp_gpu';

describe('exp_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = new Float32Array(23 * 32);
    const result = exp_gpu.uploadExpDownload(a, 23, 32);
    expect(result.length).toEqual(a.length);
  });

  it('returns e when the only value in a 1x1 matrix is 1.0', () => {
    const a = new Float32Array([1]);
    const result = exp_gpu.uploadExpDownload(a, 1, 1);
    expect(result[0]).toBeCloseTo(Math.E);
  });

  it('operates on every value in a matrix', () => {
    const a = new Float32Array([1, 1, 1, 1, 1, 1]);
    const result = exp_gpu.uploadExpDownload(a, 1, a.length);
    const expected = new Float32Array(a.length);
    expected.fill(Math.E);
    test_util.expectArraysClose(result, expected, 0.0001);
  });

  it('calculates f(x)=e^x for every value in the matrix', () => {
    const a = new Float32Array([0.5, 1, 2, -1]);
    const result = exp_gpu.uploadExpDownload(a, 1, a.length);
    const expected = new Float32Array(a.length);
    for (let i = 0; i < a.length; ++i) {
      expected[i] = Math.exp(a[i]);
    }
    test_util.expectArraysClose(result, expected, 0.0001);
  });
});
