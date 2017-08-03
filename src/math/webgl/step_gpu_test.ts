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
import * as step_gpu from './step_gpu';

describe('step_gpu', () => {
  it('returns a matrix with the shape of the input matrix', () => {
    const a = new Float32Array(67 * 901);
    const result = step_gpu.uploadStepDownload(a, 67, 901);
    expect(result.length).toEqual(a.length);
  });

  it('preserves zeroes from the input matrix', () => {
    const a = new Float32Array(1);
    const result = step_gpu.uploadStepDownload(a, 1, 1);
    expect(result[0]).toEqual(0);
  });

  it('preserves ones from the input matrix', () => {
    const a = new Float32Array([1]);
    const result = step_gpu.uploadStepDownload(a, 1, 1);
    expect(result[0]).toEqual(a[0]);
  });

  it('transforms negative values to zeroes', () => {
    const a = new Float32Array([-123.45]);
    const result = step_gpu.uploadStepDownload(a, 1, 1);
    expect(result[0]).toEqual(0);
  });

  it('transforms positive values to ones', () => {
    const a = new Float32Array([0.1]);
    const result = step_gpu.uploadStepDownload(a, 1, 1);
    expect(result[0]).toEqual(1);
  });

  it('operates on every element of a matrix', () => {
    const a = new Float32Array(24);
    a.fill(0.1);
    const result = step_gpu.uploadStepDownload(a, 4, 6);
    const expected = new Float32Array(a.length);
    expected.fill(1);
    test_util.expectArraysClose(result, expected, 0);
  });

  it('operates on a heterogeneous matrix', () => {
    const a = new Float32Array([-1, 0, 100, -0.001]);
    const result = step_gpu.uploadStepDownload(a, 4, 1);
    const expected = new Float32Array([0, 0, 1, 0]);
    test_util.expectArraysClose(result, expected, 0);
  });
});
