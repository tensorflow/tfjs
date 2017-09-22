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
import {ReduceSumProgram} from './reducesum_gpu';
import {TextureManager} from './texture_manager';

describe('reducesum_gpu', () => {
  it('returns 0 when A is [0]', () => {
    const a = new Float32Array(129 * 257);
    const result = uploadReduceSumDownload(a, 129, 257);
    expect(result).toEqual(0);
  });

  it('returns 1 when A is [1]', () => {
    const a = new Float32Array([1]);
    const result = uploadReduceSumDownload(a, 1, 1);
    expect(result).toEqual(1);
  });

  it('returns 1 when A has one 1', () => {
    const a = new Float32Array(100 * 100);
    a[49] = 1;
    const result = uploadReduceSumDownload(a, 100, 100);
    expect(result).toEqual(1);
  });

  it('returns 2 when A has two ones', () => {
    const a = new Float32Array(513 * 257);
    a[23] = 1;
    a[1000] = 1;
    const result = uploadReduceSumDownload(a, 513, 257);
    expect(result).toEqual(2);
  });

  it('accumulates values from matrix', () => {
    const a = test_util.randomArrayInRange(100, -1, 1);
    const result = uploadReduceSumDownload(a, 10, 10);
    const expected = (() => {
      let accum = 0;
      for (let i = 0; i < a.length; ++i) {
        accum += a[i];
      }
      return accum;
    })();
    expect(result).toBeCloseTo(expected);
  });

  it('computes 7 from 3x2 [1,2,3,0,0,1]', () => {
    const a = new Float32Array([1, 2, 3, 0, 0, 1]);
    const result = uploadReduceSumDownload(a, 3, 2);
    expect(result).toEqual(7);
  });

  it('sum across 2 elements', () => {
    const a = new Float32Array([3, 5]);
    const result = uploadReduceSumDownload(a, 2, 1);
    expect(result).toEqual(8);
  });

  it('sum across 3 elements', () => {
    const a = new Float32Array([3, 5, 1]);
    const result = uploadReduceSumDownload(a, 3, 1);
    expect(result).toEqual(9);
  });

  it('sum across 4 elements', () => {
    const a = new Float32Array([3, 5, 1, 2]);
    const result = uploadReduceSumDownload(a, 4, 1);
    expect(result).toEqual(11);
  });

  it('sum across 5 elements', () => {
    const a = new Float32Array([3, 5, 1, 2, 1]);
    const result = uploadReduceSumDownload(a, 5, 1);
    expect(result).toEqual(12);
  });

  it('sum across 6 elements', () => {
    const a = new Float32Array([3, 5, 1, 2, 1, -3]);
    const result = uploadReduceSumDownload(a, 6, 1);
    expect(result).toEqual(9);
  });

  it('sum across 7 elements', () => {
    const a = new Float32Array([3, 5, 1, 2, 1, -3, 5]);
    const result = uploadReduceSumDownload(a, 7, 1);
    expect(result).toEqual(14);
  });
});

export function uploadReduceSumDownload(
    a: Float32Array, rows: number, cols: number): number {
  const arr = Array2D.new([rows, cols], a);
  const out = Scalar.new(0);

  const gpgpu = new GPGPUContext();
  const textureManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, textureManager);

  const program = new ReduceSumProgram(arr.size);
  const binary = gpgpu_math.compileProgram(gpgpu, program, [arr], out);
  gpgpu_math.runProgram(binary, [arr], out);

  const result = out.get();
  arr.dispose();
  out.dispose();
  textureManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();
  return result;
}
