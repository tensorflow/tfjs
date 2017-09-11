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
import {UnaryOp} from './unaryop_gpu';
import * as unaryop_gpu_test from './unaryop_gpu_test';
import {Array2D} from '../ndarray';

describe('neg_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = new Float32Array(28 * 32);
    const result = uploadNegDownload(a, 28, 32);
    expect(result.length).toEqual(a.length);
  });

  it('preserves zero values', () => {
    const a = new Float32Array([0]);
    const result = uploadNegDownload(a, 1, 1);
    expect(result[0]).toBeCloseTo(0);
  });

  it('negates positive values into negative values', () => {
    const a = new Float32Array([1]);
    const result = uploadNegDownload(a, 1, 1);
    expect(result[0]).toEqual(-1);
  });

  it('negates negative values into positive values', () => {
    const a = new Float32Array([-1]);
    const result = uploadNegDownload(a, 1, 1);
    expect(result[0]).toEqual(1);
  });

  it('operates on every value in a matrix', () => {
    const a = new Float32Array([0.5, 0, -2.3, 4, -12, -Math.E]);
    const result = uploadNegDownload(a, 2, 3);
    const expected = new Float32Array([-0.5, 0, 2.3, -4, 12, Math.E]);
    test_util.expectArraysClose(result, expected, 0.0001);
  });
});

function uploadNegDownload(
    a: Float32Array, rows: number, cols: number): Float32Array {
  const arr = Array2D.new([rows, cols], a);
  return unaryop_gpu_test.uploadUnaryDownload(arr, UnaryOp.NEG);
}
