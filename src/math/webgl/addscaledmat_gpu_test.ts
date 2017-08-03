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
import * as addscaledmat_gpu from './addscaledmat_gpu';

function cpuAddScaledMatrices(
    a: Float32Array, aScalar: number, b: Float32Array,
    bScalar: number): Float32Array {
  const result = new Float32Array(a.length);
  for (let i = 0; i < result.length; ++i) {
    result[i] = (a[i] * aScalar) + (b[i] * bScalar);
  }
  return result;
}

describe('addscaledmat_gpu', () => {
  it('returns a matrix with the same shape as the input matrices', () => {
    const a = new Float32Array(9 * 14);
    const b = new Float32Array(a.length);
    const result =
        addscaledmat_gpu.uploadAddScaledMatricesDownload(a, b, 9, 14, 0, 0);
    expect(result.length).toEqual(9 * 14);
  });

  it('returns A + B when scalars are 1', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    const result =
        addscaledmat_gpu.uploadAddScaledMatricesDownload(a, b, 3, 2, 1, 1);
    test_util.expectArraysClose(
        result, new Float32Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]), 0.0001);
  });

  it('returns A * aScalar when B and bScalar are 0', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array(a.length);
    const result =
        addscaledmat_gpu.uploadAddScaledMatricesDownload(a, b, 3, 2, 1.1, 0);
    test_util.expectArraysClose(
        result, new Float32Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]), 0.0001);
  });

  it('returns B * bScalar when A and aScalar are 0', () => {
    const b = new Float32Array([1, 2, 3, 4, 5, 6]);
    const a = new Float32Array(b.length);
    const result =
        addscaledmat_gpu.uploadAddScaledMatricesDownload(a, b, 3, 2, 0, 1.1);
    test_util.expectArraysClose(
        result, new Float32Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]), 0.0001);
  });

  it('returns (A * aScalar) + (B * bScalar)', () => {
    const a = test_util.randomArrayInRange(12 * 12, -2, 2);
    const b = test_util.randomArrayInRange(a.length, -10, 10);
    const aScalar = 0.5;
    const bScalar = 0.25;
    const result = addscaledmat_gpu.uploadAddScaledMatricesDownload(
        a, b, 12, 12, aScalar, bScalar);
    test_util.expectArraysClose(
        result, cpuAddScaledMatrices(a, aScalar, b, bScalar), 0.001);
  });
});
