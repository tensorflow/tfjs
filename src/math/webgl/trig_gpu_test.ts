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
import * as trig_gpu from './trig_gpu';

describe('sin_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = new Float32Array(28 * 32);
    const result = trig_gpu.uploadSinDownload(a, 28, 32);
    expect(result.length).toEqual(a.length);
  });

  it('Sin equals CPU', () => {
    const size = 10;
    const a = test_util.randomArrayInRange(size, -1, 1);
    const expectedResult = new Float32Array(size);
    for (let i = 0; i < a.length; i++) {
      expectedResult[i] = Math.sin(a[i]);
    }
    const result = trig_gpu.uploadSinDownload(a, 1, size);
    test_util.expectArraysClose(result, expectedResult, 1e-3);
  });
});


describe('tanh_gpu', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = new Float32Array(28 * 32);
    const result = trig_gpu.uploadTanhDownload(a, 28, 32);
    expect(result.length).toEqual(a.length);
  });

  it('Tanh equals CPU', () => {
    const size = 10;
    const a = test_util.randomArrayInRange(size, -1, 1);
    const expectedResult = new Float32Array(size);
    for (let i = 0; i < a.length; i++) {
      expectedResult[i] = util.tanh(a[i]);
    }
    const result = trig_gpu.uploadTanhDownload(a, 1, size);
    test_util.expectArraysClose(result, expectedResult, 1e-6);
  });
});
