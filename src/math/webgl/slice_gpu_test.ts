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

import {Array1D, Array2D, Array3D, Array4D, initializeGPU} from '../ndarray';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {SliceProgram} from './slice_gpu';
import {TextureManager} from './texture_manager';
import * as webgl_util from './webgl_util';

describe('slice1d_gpu', () => {
  let gpgpu: GPGPUContext;
  let texManager: TextureManager;

  beforeAll(() => {
    gpgpu = new GPGPUContext();
    texManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, texManager);
  });

  afterAll(() => {
    texManager.dispose();
    gpgpu.dispose();
  });

  it('slices 1x1 into 1x1 (effectively a copy)', () => {
    const a = Array1D.new([5]);
    const result = doSlice1D(a, 0, 1);
    expect(result.shape).toEqual([1]);
    expect(result.get(0)).toBe(5);
  });

  it('slices 5x1 into shape 2x1 starting at 3', () => {
    const a = Array1D.new([1, 2, 3, 4, 5]);
    const result = doSlice1D(a, 3, 2);
    expect(result.shape).toEqual([2]);
    expect(result.getValues()).toEqual(new Float32Array([4, 5]));
  });

  it('slices 5x1 into shape 3x1 starting at 1', () => {
    const a = Array1D.new([1, 2, 3, 4, 5]);
    const result = doSlice1D(a, 1, 3);
    expect(result.shape).toEqual([3]);
    expect(result.getValues()).toEqual(new Float32Array([2, 3, 4]));
  });

  it('slices array that is bigger than max tex size', () => {
    const maxTexSize = webgl_util.queryMaxTextureSize(gpgpu.gl);
    const a = Array1D.randUniform([maxTexSize + 10], -1, 1);
    const expected = a.get(a.size - 1);
    const result = doSlice1D(a, a.size - 1, 1);
    expect(result.shape).toEqual([1]);
    expect(result.get(0)).toEqual(expected);
  });

  function doSlice1D(a: Array1D, start: number, size: number): Array1D {
    const program = new SliceProgram([size]);
    const result = Array1D.zeros([size]);

    const binary = gpgpu_math.compileProgram(gpgpu, program, [a], result);
    const customSetup = program.getCustomSetupFunc([start]);
    gpgpu_math.runProgram(binary, [a], result, customSetup);

    a.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);

    return result;
  }
});

describe('slice2d_gpu', () => {
  let gpgpu: GPGPUContext;
  let texManager: TextureManager;

  beforeAll(() => {
    gpgpu = new GPGPUContext();
    texManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, texManager);
  });

  afterAll(() => {
    texManager.dispose();
    gpgpu.dispose();
  });

  it('slices 1x1 into shape 1x1 (effectively a copy)', () => {
    const a = Array2D.new([1, 1], [[5]]);
    const result = doSlice2D(a, [0, 0], [1, 1]);
    expect(result.shape).toEqual([1, 1]);
    expect(result.get(0, 0)).toBe(5);
  });

  it('slices 3x3 array into 2x2 starting at [1, 1]', () => {
    const a = Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const result = doSlice2D(a, [1, 1], [2, 2]);
    expect(result.shape).toEqual([2, 2]);
    expect(result.getValues()).toEqual(new Float32Array([5, 6, 8, 9]));
  });

  it('slices 3x3 into 2x1 starting at [1,1]', () => {
    const a = Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const result = doSlice2D(a, [1, 1], [2, 1]);
    expect(result.shape).toEqual([2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([5, 8]));
  });

  it('slices array that is bigger than max tex size', () => {
    const maxTexSize = webgl_util.queryMaxTextureSize(gpgpu.gl);
    const a = Array2D.randUniform([maxTexSize + 10, 1], -1, 1);
    const expected = a.get(a.size - 1, 0);
    const result = doSlice2D(a, [a.size - 1, 0], [1, 1]);
    expect(result.shape).toEqual([1, 1]);
    expect(result.get(0, 0)).toEqual(expected);
  });

  function doSlice2D(
      a: Array2D, start: [number, number], size: [number, number]): Array2D {
    const program = new SliceProgram(size);
    const result = Array2D.zeros(size);

    const binary = gpgpu_math.compileProgram(gpgpu, program, [a], result);
    const customSetup = program.getCustomSetupFunc(start);
    gpgpu_math.runProgram(binary, [a], result, customSetup);

    a.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);

    return result;
  }
});

describe('slice3d_gpu', () => {
  let gpgpu: GPGPUContext;
  let texManager: TextureManager;

  beforeAll(() => {
    gpgpu = new GPGPUContext();
    texManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, texManager);
  });

  afterAll(() => {
    texManager.dispose();
    gpgpu.dispose();
  });

  it('slices 1x1x1 into shape 1x1x1 (effectively a copy)', () => {
    const a = Array3D.new([1, 1, 1], [[[5]]]);
    const result = doSlice3D(a, [0, 0, 0], [1, 1, 1]);
    expect(result.shape).toEqual([1, 1, 1]);
    expect(result.get(0, 0, 0)).toBe(5);
  });

  it('slices 2x2x2 array into 1x2x2 starting at [1, 0, 0]', () => {
    const a = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
    const result = doSlice3D(a, [1, 0, 0], [1, 2, 2]);
    expect(result.shape).toEqual([1, 2, 2]);
    expect(result.getValues()).toEqual(new Float32Array([5, 6, 7, 8]));
  });

  it('slices 2x2x2 array into 2x1x1 starting at [0, 1, 1]', () => {
    const a = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
    const result = doSlice3D(a, [0, 1, 1], [2, 1, 1]);
    expect(result.shape).toEqual([2, 1, 1]);
    expect(result.getValues()).toEqual(new Float32Array([4, 8]));
  });

  it('slices array that is bigger than max tex size', () => {
    const maxTexSize = webgl_util.queryMaxTextureSize(gpgpu.gl);
    const a = Array3D.randUniform([maxTexSize + 10, 1, 1], -1, 1);
    const expected = a.get(a.size - 1, 0, 0);
    const result = doSlice3D(a, [a.size - 1, 0, 0], [1, 1, 1]);
    expect(result.shape).toEqual([1, 1, 1]);
    expect(result.get(0, 0, 0)).toEqual(expected);
  });

  function doSlice3D(
      a: Array3D, start: [number, number, number],
      size: [number, number, number]): Array3D {
    const program = new SliceProgram(size);
    const result = Array3D.zeros(size);

    const binary = gpgpu_math.compileProgram(gpgpu, program, [a], result);
    const customSetup = program.getCustomSetupFunc(start);
    gpgpu_math.runProgram(binary, [a], result, customSetup);

    a.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);

    return result;
  }
});

describe('slice4d_gpu', () => {
  let gpgpu: GPGPUContext;
  let texManager: TextureManager;

  beforeAll(() => {
    gpgpu = new GPGPUContext();
    texManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, texManager);
  });

  afterAll(() => {
    texManager.dispose();
    gpgpu.dispose();
  });

  it('slices 1x1x1x1 into shape 1x1x1x1 (effectively a copy)', () => {
    const a = Array4D.new([1, 1, 1, 1], [[[[5]]]]);
    const result = doSlice4D(a, [0, 0, 0, 0], [1, 1, 1, 1]);
    expect(result.shape).toEqual([1, 1, 1, 1]);
    expect(result.get(0, 0, 0, 0)).toBe(5);
  });

  it('slices 2x2x2x2 array into 1x2x2x2 starting at [1, 0, 0, 0]', () => {
    const a = Array4D.new(
        [2, 2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88]);
    const result = doSlice4D(a, [1, 0, 0, 0], [1, 2, 2, 2]);
    expect(result.shape).toEqual([1, 2, 2, 2]);
    expect(result.getValues()).toEqual(new Float32Array([
      11, 22, 33, 44, 55, 66, 77, 88
    ]));
  });

  it('slices 2x2x2x2 array into 2x1x1x1 starting at [0, 1, 1, 1]', () => {
    const a = Array4D.new(
        [2, 2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88]);
    const result = doSlice4D(a, [0, 1, 1, 1], [2, 1, 1, 1]);
    expect(result.shape).toEqual([2, 1, 1, 1]);
    expect(result.getValues()).toEqual(new Float32Array([8, 88]));
  });

  it('slices array that is bigger than max tex size', () => {
    const maxTexSize = webgl_util.queryMaxTextureSize(gpgpu.gl);
    const a = Array4D.randUniform([maxTexSize + 10, 1, 1, 1], -1, 1);
    const expected = a.get(a.size - 1, 0, 0, 0);
    const result = doSlice4D(a, [a.size - 1, 0, 0, 0], [1, 1, 1, 1]);
    expect(result.shape).toEqual([1, 1, 1, 1]);
    expect(result.get(0, 0, 0, 0)).toEqual(expected);
  });

  function doSlice4D(
      a: Array4D, start: [number, number, number, number],
      size: [number, number, number, number]): Array4D {
    const program = new SliceProgram(size);
    const result = Array4D.zeros(size);

    const binary = gpgpu_math.compileProgram(gpgpu, program, [a], result);
    const customSetup = program.getCustomSetupFunc(start);
    gpgpu_math.runProgram(binary, [a], result, customSetup);

    a.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);

    return result;
  }
});
