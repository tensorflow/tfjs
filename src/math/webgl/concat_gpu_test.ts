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
// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, Array3D, Array4D, initializeGPU, NDArray} from '../ndarray';
import {ConcatProgram} from './concat_gpu';
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
    const program = new ConcatProgram(a.shape, b.shape, 0);
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
    const program = new ConcatProgram(a.shape, b.shape, axis);
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

describe('concat3d_gpu', () => {
  it('concat axis=0', () => {
    const x1 = new Float32Array([1, 11, 111, 2, 22, 222]);
    const x2 =
        new Float32Array([5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);

    const result = uploadConcat3dDownload(x1, x2, [1, 2, 3], [2, 2, 3], 0);
    test_util.expectArraysClose(
        result, new Float32Array([
          1, 11, 111, 2, 22, 222, 5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888
        ]),
        1e-6);

  });

  it('concat axis=1', () => {
    const x1 = new Float32Array([1, 11, 111, 3, 33, 333]);
    const x2 =
        new Float32Array([5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);

    const result = uploadConcat3dDownload(x1, x2, [2, 1, 3], [2, 2, 3], 1);
    test_util.expectArraysClose(
        result, new Float32Array([
          1, 11, 111, 5, 55, 555, 6, 66, 666, 3, 33, 333, 7, 77, 777, 8, 88, 888
        ]),
        1e-6);
  });

  it('concat axis=2', () => {
    const x1 = new Float32Array([1, 11, 2, 22, 3, 33, 4, 44]);
    const x2 =
        new Float32Array([5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);

    const result = uploadConcat3dDownload(x1, x2, [2, 2, 2], [2, 2, 3], 2);
    test_util.expectArraysClose(
        result, new Float32Array([
          1, 11, 5, 55, 555, 2, 22, 6, 66, 666,
          3, 33, 7, 77, 777, 4, 44, 8, 88, 888
        ]),
        1e-6);
  });
});

function uploadConcat3dDownload(
    a: Float32Array, b: Float32Array, aShape: [number, number, number],
    bShape: [number, number, number], axis: number): Float32Array {
  const gpgpu = new GPGPUContext();
  gpgpu.enableAutomaticDebugValidation(true);
  const textureManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, textureManager);

  const program = new ConcatProgram(aShape, bShape, axis);
  const aArr = Array3D.new(aShape, a);
  const bArr = Array3D.new(bShape, b);
  const rArr = NDArray.zeros(program.outputShape);
  const binary = gpgpu_math.compileProgram(gpgpu, program, [aArr, bArr], rArr);
  gpgpu_math.runProgram(binary, [aArr, bArr], rArr);
  const result = rArr.getValues();

  aArr.dispose();
  bArr.dispose();
  rArr.dispose();
  textureManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();

  return result;
}

describe('concat4d_gpu', () => {
  it('concat axis=0', () => {
    const x1 = Array4D.new([1, 2, 3, 1], [1, 11, 111, 2, 22, 222]);
    const x2 = Array4D.new(
        [2, 2, 3, 1], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);

    const result = doConcat4D(x1, x2, 0);
    test_util.expectArraysClose(
        result, new Float32Array([
          1, 11, 111, 2, 22, 222, 5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888
        ]),
        1e-6);

  });

  it('concat axis=1', () => {
    const x1 = Array4D.new([2, 1, 3, 1], [1, 11, 111, 3, 33, 333]);
    const x2 = Array4D.new(
        [2, 2, 3, 1], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);

    const result = doConcat4D(x1, x2, 1);
    test_util.expectArraysClose(
        result, new Float32Array([
          1, 11, 111, 5, 55, 555, 6, 66, 666, 3, 33, 333, 7, 77, 777, 8, 88, 888
        ]),
        1e-6);
  });

  it('concat axis=2', () => {
    const x1 = Array4D.new([2, 2, 2, 1], [1, 11, 2, 22, 3, 33, 4, 44]);
    const x2 = Array4D.new(
        [2, 2, 3, 1], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);

    const result = doConcat4D(x1, x2, 2);
    test_util.expectArraysClose(
        result, new Float32Array([
          1, 11, 5, 55, 555, 2, 22, 6, 66, 666,
          3, 33, 7, 77, 777, 4, 44, 8, 88, 888
        ]),
        1e-6);
  });

  function doConcat4D(a: Array4D, b: Array4D, axis: number): Float32Array {
    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    const textureManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, textureManager);

    const program = new ConcatProgram(a.shape, b.shape, axis);
    const rArr =
        Array4D.zeros(program.outputShape as [number, number, number, number]);
    const binary = gpgpu_math.compileProgram(gpgpu, program, [a, b], rArr);
    gpgpu_math.runProgram(binary, [a, b], rArr);
    const result = rArr.getValues();

    a.dispose();
    b.dispose();
    rArr.dispose();
    textureManager.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();

    return result;
  }
});
