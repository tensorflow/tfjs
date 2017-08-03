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
import * as mulbcast_gpu from './mulbcast_gpu';

export function cpuMultiplyBroadcast(
    a: Float32Array, aNumRows: number, aNumCols: number, b: Float32Array,
    bNumRows: number, bNumCols: number): Float32Array {
  const resultNumRows = Math.max(aNumRows, bNumRows);
  const resultNumCols = Math.max(aNumCols, bNumCols);
  const result = new Float32Array(resultNumRows * resultNumCols);
  let dst = 0;
  for (let r = 0; r < resultNumRows; ++r) {
    for (let c = 0; c < resultNumCols; ++c) {
      const ai = ((r % aNumRows) * aNumCols) + (c % aNumCols);
      const bi = ((r % bNumRows) * bNumCols) + (c % bNumCols);
      result[dst] = a[ai] * b[bi];
      ++dst;
    }
  }
  return result;
}

describe('mulbcast_gpu', () => {
  it('returns a matrix dimensions [max(aRows, bRows), max(aCols, bCols)]',
     () => {
       const a = new Float32Array(13 * 100);
       const b = new Float32Array(100 * 99);
       const result =
           mulbcast_gpu.uploadMultiplyBroadcastDownload(a, 1, 100, b, 100, 1);
       expect(result.length).toEqual(100 * 100);
     });

  it('returns [0] when A is [0], A and B same size', () => {
    const a = new Float32Array(16 * 16);
    const b = test_util.randomArrayInRange(16 * 16, -10, 10);
    const result =
        mulbcast_gpu.uploadMultiplyBroadcastDownload(a, 16, 16, b, 16, 16);
    test_util.expectArraysClose(a, result, 0.00001);
  });

  it('returns [0] when B is [0], A and B same size', () => {
    const a = test_util.randomArrayInRange(16 * 16, -10, 10);
    const b = new Float32Array(16 * 16);
    const result =
        mulbcast_gpu.uploadMultiplyBroadcastDownload(a, 16, 16, b, 16, 16);
    test_util.expectArraysClose(b, result, 0.00001);
  });

  it('returns A when B is [1] and matrices have the same size', () => {
    const a = new Float32Array(16 * 16);
    a.fill(1);
    const b = test_util.randomArrayInRange(16 * 16, -10, 10);
    const result =
        mulbcast_gpu.uploadMultiplyBroadcastDownload(a, 16, 16, b, 16, 16);
    test_util.expectArraysClose(result, b, 0.00001);
  });

  it('returns B when A is [1] and matrices have the same size', () => {
    const a = test_util.randomArrayInRange(16 * 16, -10, 10);
    const b = new Float32Array(16 * 16);
    b.fill(1);
    const result =
        mulbcast_gpu.uploadMultiplyBroadcastDownload(a, 16, 16, b, 16, 16);
    test_util.expectArraysClose(result, a, 0.00001);
  });

  it('returns B when A is [1] and A is narrower than B', () => {
    const a = new Float32Array(16 * 8);
    a.fill(1);
    const b = test_util.randomArrayInRange(16 * 16, -10, 10);
    const result =
        mulbcast_gpu.uploadMultiplyBroadcastDownload(a, 16, 8, b, 16, 16);
    test_util.expectArraysClose(result, b, 0.00001);
  });

  it('returns B when A is [1] and A is shorter than B', () => {
    const a = new Float32Array(8 * 16);
    a.fill(1);
    const b = test_util.randomArrayInRange(16 * 16, -10, 10);
    const result =
        mulbcast_gpu.uploadMultiplyBroadcastDownload(a, 8, 16, b, 16, 16);
    test_util.expectArraysClose(result, b, 0.00001);
  });

  it('returns B when A is [1] and A is smaller than B', () => {
    const a = new Float32Array(7 * 6);
    a.fill(1);
    const b = test_util.randomArrayInRange(18 * 21, -1, 1);
    const result =
        mulbcast_gpu.uploadMultiplyBroadcastDownload(a, 7, 6, b, 18, 21);
    test_util.expectArraysClose(result, b, 0.00001);
  });

  it('broadcasts a smaller A [2x2] across B [4x4]', () => {
    const a = new Float32Array([1, 0, 1, 0]);
    const b = new Float32Array(4 * 4);
    for (let i = 0; i < b.length; ++i) {
      b[i] = i + 1;
    }
    const expected =
        new Float32Array([1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0]);
    const gpuResult =
        mulbcast_gpu.uploadMultiplyBroadcastDownload(a, 2, 2, b, 4, 4);
    const cpuResult = cpuMultiplyBroadcast(a, 2, 2, b, 4, 4);
    test_util.expectArraysClose(cpuResult, expected, 0.0001);
    test_util.expectArraysClose(gpuResult, expected, 0.0001);
  });

  it('broadcasts a non-square A [3x5] across a larger B [16x16]', () => {
    const a = test_util.randomArrayInRange(3 * 5, -1, 1);
    const b = test_util.randomArrayInRange(16 * 16, -1, 1);
    const result =
        mulbcast_gpu.uploadMultiplyBroadcastDownload(a, 3, 5, b, 16, 16);
    test_util.expectArraysClose(
        result, cpuMultiplyBroadcast(a, 3, 5, b, 16, 16), 0.0001);
  });

  it('broadcasts a non-square A across a larger non-square B', () => {
    const a = test_util.randomArrayInRange(37 * 63, -1, 1);
    const b = test_util.randomArrayInRange(128 * 150, -1, 1);
    const result =
        mulbcast_gpu.uploadMultiplyBroadcastDownload(a, 37, 63, b, 128, 150);
    test_util.expectArraysClose(
        result, cpuMultiplyBroadcast(a, 37, 63, b, 128, 150), 0.0001);
  });
});
