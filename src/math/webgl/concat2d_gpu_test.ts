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
import {Array2D, initializeGPU} from '../ndarray';
import {Concat2DProgram} from './concat2d_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {TextureManager} from './texture_manager';
import * as webgl_util from './webgl_util';

describe('concat2d_gpu', () => {
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

  it('[[3]] + [[5]], axis=0', () => {
    const axis = 0;
    const a = Array2D.new([1, 1], [3]);
    const b = Array2D.new([1, 1], [5]);

    const result = doConcat(a, b, axis);
    const expected = new Float32Array([3, 5]);

    expect(result.shape).toEqual([2, 1]);
    test_util.expectArraysClose(result.getValues(), expected);
  });

  it('[[3]] + [[5]], axis=1', () => {
    const axis = 1;
    const a = Array2D.new([1, 1], [3]);
    const b = Array2D.new([1, 1], [5]);

    const result = doConcat(a, b, axis);
    const expected = new Float32Array([3, 5]);

    expect(result.shape).toEqual([1, 2]);
    test_util.expectArraysClose(result.getValues(), expected);
  });

  it('[[1, 2], [3, 4]] + [[5, 6]], axis=0', () => {
    const axis = 0;
    const a = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    const b = Array2D.new([1, 2], [[5, 6]]);

    const result = doConcat(a, b, axis);
    const expected = new Float32Array([1, 2, 3, 4, 5, 6]);

    expect(result.shape).toEqual([3, 2]);
    test_util.expectArraysClose(result.getValues(), expected);
  });

  it('[[1, 2], [3, 4]] + [[5, 6], [7, 8]], axis=1', () => {
    const axis = 1;
    const a = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    const b = Array2D.new([2, 2], [[5, 6], [7, 8]]);

    const result = doConcat(a, b, axis);
    const expected = new Float32Array([1, 2, 5, 6, 3, 4, 7, 8]);

    expect(result.shape).toEqual([2, 4]);
    test_util.expectArraysClose(result.getValues(), expected);
  });

  it('matches cpu with arrays bigger than max tex size, axis=0', () => {
    const axis = 0;
    const maxTextureSize = webgl_util.queryMaxTextureSize(gpgpu.gl);
    const a = Array2D.randUniform([maxTextureSize + 10, 1], -1, 1);
    const b = Array2D.randUniform([maxTextureSize + 10, 1], -1, 1);

    const result = doConcat(a, b, axis, false);
    const expected = doCpuConcat(a, b, axis).getValues();
    const resVals = result.getValues();

    expect(result.shape).toEqual([maxTextureSize * 2 + 20, 1]);
    test_util.expectArraysClose(resVals, expected);

    a.dispose();
    b.dispose();
    result.dispose();
  });

  it('matches cpu with arrays bigger than max tex size, axis=1', () => {
    const axis = 1;
    const maxTextureSize = webgl_util.queryMaxTextureSize(gpgpu.gl);
    const a = Array2D.randUniform([maxTextureSize + 10, 1], -1, 1);
    const b = Array2D.randUniform([maxTextureSize + 10, 1], -1, 1);

    const result = doConcat(a, b, axis, false);
    const expected = doCpuConcat(a, b, axis).getValues();
    const resVals = result.getValues();

    expect(result.shape).toEqual([maxTextureSize + 10, 2]);
    test_util.expectArraysClose(resVals, expected);

    a.dispose();
    b.dispose();
    result.dispose();
  });

  function doCpuConcat(a: Array2D, b: Array2D, axis: number): Array2D {
    const mathCpu = new NDArrayMathCPU();
    return mathCpu.concat2D(a, b, axis);
  }

  function doConcat(
      a: Array2D, b: Array2D, axis: number, dispose = true): Array2D {
    const program = new Concat2DProgram(a.shape, b.shape, axis);
    const r = Array2D.zeros(program.outputShape as [number, number]);
    const binary = gpgpu_math.compileProgram(gpgpu, program, [a, b], r);
    gpgpu_math.runProgram(binary, [a, b], r);

    if (dispose) {
      a.dispose();
      b.dispose();
    }
    gpgpu.deleteProgram(binary.webGLProgram);

    return r;
  }
});
