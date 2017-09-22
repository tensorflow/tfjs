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
import {Array2D, initializeGPU, Scalar} from '../ndarray';

import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {LogSumExpProgram} from './logsumexp_gpu';
import {TextureManager} from './texture_manager';

function cpuLogSumExp(m: Float32Array): number {
  if (m.length === 0) {
    throw new Error('m must have length greater than zero.');
  }
  let mMax = m[0];
  for (let i = 0; i < m.length; ++i) {
    mMax = Math.max(mMax, m[i]);
  }
  let expSum = 0;
  for (let i = 0; i < m.length; ++i) {
    expSum += Math.exp(m[i] - mMax);
  }
  const logSumExp = mMax + Math.log(expSum);
  return logSumExp;
}

describe('logsumexp_gpu', () => {
  it('logsumexp(1) = 0', () => {
    const a = new Float32Array([0]);
    const result = uploadLogSumExpDownload(a, 1, 1);
    expect(result).toEqual(0);
  });

  it('returns ln(length) when the input matrix is [0]', () => {
    const a = new Float32Array(512 * 512);
    const result = uploadLogSumExpDownload(a, 512, 512);
    expect(result).toBeCloseTo(Math.log(a.length));
  });

  it('same as cpuLogSumExp, 1 element', () => {
    const a = test_util.randomArrayInRange(1, -2, 2);
    const result = uploadLogSumExpDownload(a, 1, 1);
    const expected = cpuLogSumExp(a);
    expect(result).toBeCloseTo(expected);
  });

  it('same as cpuLogSumExp, 2 elements', () => {
    const a = test_util.randomArrayInRange(2, -2, 2);
    const result = uploadLogSumExpDownload(a, 2, 1);
    const expected = cpuLogSumExp(a);
    expect(result).toBeCloseTo(expected);
  });

  it('same as cpuLogSumExp, 3 elements', () => {
    const a = test_util.randomArrayInRange(3, -2, 2);
    const result = uploadLogSumExpDownload(a, 3, 1);
    const expected = cpuLogSumExp(a);
    expect(result).toBeCloseTo(expected);
  });

  it('same as cpuLogSumExp, 4 elements', () => {
    const a = test_util.randomArrayInRange(4, -2, 2);
    const result = uploadLogSumExpDownload(a, 4, 1);
    const expected = cpuLogSumExp(a);
    expect(result).toBeCloseTo(expected);
  });

  it('same as cpuLogSumExp, 9 elements, last is max', () => {
    const a = test_util.randomArrayInRange(9, -2, 2);
    a[a.length - 1] = 3;
    const result = uploadLogSumExpDownload(a, 9, 1);
    const expected = cpuLogSumExp(a);
    expect(result).toBeCloseTo(expected);
  });

  it('same as cpuLogSumExp, 10 elements, last is max', () => {
    const a = test_util.randomArrayInRange(10, -2, 2);
    a[a.length - 1] = 3;
    const result = uploadLogSumExpDownload(a, 10, 1);
    const expected = cpuLogSumExp(a);
    expect(result).toBeCloseTo(expected);
  });

  it('same as cpuLogSumExp, 11 elements, last is max', () => {
    const a = test_util.randomArrayInRange(11, -2, 2);
    a[a.length - 1] = 3;
    const result = uploadLogSumExpDownload(a, 11, 1);
    const expected = cpuLogSumExp(a);
    expect(result).toBeCloseTo(expected);
  });

  it('same as cpuLogSumExp many elements', () => {
    const a = test_util.randomArrayInRange(12 * 29, -2, 2);
    const result = uploadLogSumExpDownload(a, 12, 29);
    const expected = cpuLogSumExp(a);
    expect(result).toBeCloseTo(expected);
  });
});

export function uploadLogSumExpDownload(
    a: Float32Array, rows: number, columns: number): number {
  const gpgpu = new GPGPUContext();
  const textureManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, textureManager);
  const aArr = Array2D.new([rows, columns], a);
  const rScalar = Scalar.new(0);
  const program = new LogSumExpProgram(aArr.size);
  const binary = gpgpu_math.compileProgram(gpgpu, program, [aArr], rScalar);
  gpgpu_math.runProgram(binary, [aArr], rScalar);
  const result = rScalar.get();
  textureManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();
  return result;
}
