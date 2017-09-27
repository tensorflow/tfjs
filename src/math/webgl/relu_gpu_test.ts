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
import {Array1D, Array2D, Array3D, NDArray, Scalar} from '../ndarray';

import * as unaryop_gpu from './unaryop_gpu';
import * as unaryop_gpu_test from './unaryop_gpu_test';

describe('relu_gpu', () => {
  it('returns a matrix with the shape of the input matrix', () => {
    const a = Array2D.zeros([17, 257]);
    const result = uploadReluDownload(a);
    expect(result.length).toEqual(a.size);
  });

  it('does nothing to positive values', () => {
    const a = Array1D.new([1]);
    const result = uploadReluDownload(a);
    expect(result[0]).toEqual(1);
  });

  it('sets negative values to 0', () => {
    const a = Array1D.new([-1]);
    const result = uploadReluDownload(a);
    expect(result[0]).toEqual(0);
  });

  it('preserves zero values', () => {
    const a = Scalar.new(0);
    const result = uploadReluDownload(a);
    expect(result[0]).toEqual(0);
  });

  it('operates on multiple values', () => {
    const a = Array2D.new([3, 3], [[-1, 2, -3], [4, -5, 6], [-7, 8, -9]]);
    const result = uploadReluDownload(a);
    test_util.expectArraysClose(
        result, new Float32Array([0, 2, 0, 4, 0, 6, 0, 8, 0]));
  });

  it('propagates NaNs', () => {
    const a = Array3D.new([2, 2, 2], [-1, NaN, -3, 4, 6, 0, -3, 1]);
    const result = uploadReluDownload(a);
    test_util.expectArraysClose(
        result, new Float32Array([0, NaN, 0, 4, 6, 0, 0, 1]));
  });
});

function uploadReluDownload(a: NDArray): Float32Array {
  return unaryop_gpu_test.uploadUnaryDownload(a, unaryop_gpu.RELU);
}
