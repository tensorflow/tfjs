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
import {ReduceSumProgram} from './reducesum_gpu';
import {GPGPUContext} from './gpgpu_context';
import {Array2D, initializeGPU, Scalar} from '../ndarray';
import {TextureManager} from './texture_manager';
import * as gpgpu_math from './gpgpu_math';

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
});

export function uploadReduceSumDownload(a: Float32Array, rows: number,
    cols: number): number {
  const arr = Array2D.new([rows, cols], a);
  const out = Scalar.new(0);

  const gpgpu = new GPGPUContext();
  const textureManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, textureManager);

  const program = new ReduceSumProgram(arr.size);
  const binary = gpgpu_math.compileProgram(gpgpu, program, [arr], out);
  gpgpu_math.runProgram(binary);

  const result = out.get();
  arr.dispose();
  out.dispose();
  textureManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();
  return result;
}
