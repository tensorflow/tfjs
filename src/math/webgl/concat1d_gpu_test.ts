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
import {NDArrayMathCPU} from '../math_cpu';
import {Array1D, initializeGPU} from '../ndarray';
import {Concat1DProgram} from './concat1d_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {TextureManager} from './texture_manager';
import * as webgl_util from './webgl_util';

describe('concat1d_gpu', () => {
  let gpgpu: GPGPUContext;
  let textureManager: TextureManager;

  beforeAll(() => {
    gpgpu = new GPGPUContext();
    textureManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, textureManager);
  });

  afterAll(() => {
    textureManager.dispose();
    gpgpu.dispose();
  });

  it('3 + 5', () => {
    const a = Array1D.new([3]);
    const b = Array1D.new([5]);

    const result = doConcat(a, b);
    const expected = new Float32Array([3, 5]);
    test_util.expectArraysClose(result, expected);
  });

  it('3 + [5,7]', () => {
    const a = Array1D.new([3]);
    const b = Array1D.new([5, 7]);

    const result = doConcat(a, b);
    const expected = new Float32Array([3, 5, 7]);
    test_util.expectArraysClose(result, expected);
  });

  it('[3,5] + 7', () => {
    const a = Array1D.new([3, 5]);
    const b = Array1D.new([7]);

    const result = doConcat(a, b);
    const expected = new Float32Array([3, 5, 7]);
    test_util.expectArraysClose(result, expected);
  });

  it('matches cpu with arrays bigger than max tex size', () => {
    const maxTextureSize = webgl_util.queryMaxTextureSize(gpgpu.gl);
    const a = Array1D.randUniform([maxTextureSize + 10], -1, 1);
    const b = Array1D.randUniform([maxTextureSize + 10], -1, 1);

    const result = doConcat(a, b, false);
    const expected = doCpuConcat(a, b);
    a.dispose();
    b.dispose();

    test_util.expectArraysClose(result, expected);
  });

  function doCpuConcat(a: Array1D, b: Array1D): Float32Array {
    const mathCpu = new NDArrayMathCPU();
    return mathCpu.concat1D(a, b).getValues();
  }

  function doConcat(a: Array1D, b: Array1D, dispose = true): Float32Array {
    const program = new Concat1DProgram(a.size, b.size);
    const r = Array1D.zeros(program.outputShape as [number]);
    const binary = gpgpu_math.compileProgram(gpgpu, program, [a, b], r);
    gpgpu_math.runProgram(binary, [a, b], r);
    const result = r.getValues();

    if (dispose) {
      a.dispose();
      b.dispose();
    }
    r.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);

    return result;
  }
});
