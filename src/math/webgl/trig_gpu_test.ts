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
import * as util from '../../util';
import {UnaryOp} from './unaryop_gpu';
import * as unaryop_gpu_test from './unaryop_gpu_test';
import {Array1D, Array2D, Array3D} from '../ndarray';

describe('sin_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array2D.zeros([28, 28]);
    const result = unaryop_gpu_test.uploadUnaryDownload(a, UnaryOp.SIN);
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
    const result = unaryop_gpu_test.uploadUnaryDownload(aArr, UnaryOp.SIN);
    test_util.expectArraysClose(result, expectedResult, 1e-3);
  });
});


describe('tanh_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array3D.zeros([28, 14, 2]);
    const result = unaryop_gpu_test.uploadUnaryDownload(a, UnaryOp.TANH);
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
    const result = unaryop_gpu_test.uploadUnaryDownload(aArr, UnaryOp.TANH);
    test_util.expectArraysClose(result, expectedResult, 1e-6);
  });
});
