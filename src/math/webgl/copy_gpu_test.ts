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
import {Array2D, initializeGPU} from '../ndarray';
import {Copy2DProgram} from './copy_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {TextureManager} from './texture_manager';

function uploadCopyDownload(
    srcVals: Float32Array, srcShape: [number, number],
    srcStart: [number, number], srcSize: [number, number],
    destStart: [number, number], destSize: [number, number],
    destVals: Float32Array, destShape: [number, number]): Float32Array {
  const gpgpu = new GPGPUContext();
  const texManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, texManager);

  const program = new Copy2DProgram(srcSize[1], destSize[1]);
  const source = Array2D.new(srcShape, srcVals);
  const dest = Array2D.new(destShape, destVals);

  const binary = gpgpu_math.compileProgram(gpgpu, program, [source], dest);
  const customSetup = program.getCustomSetupFunc(srcStart, destStart, destSize);
  gpgpu_math.runProgram(binary, [source], dest, customSetup);
  const result = dest.getValues();

  source.dispose();
  dest.dispose();
  texManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();

  return result;
}

describe('copy_gpu', () => {
  it('copies a 1x1 source to a 1x1 dest', () => {
    const source = new Float32Array([Math.PI]);
    const dest = new Float32Array([0]);
    const result = uploadCopyDownload(
        source, [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], dest, [1, 1]);
    expect(result.length).toEqual(1);
    expect(result[0]).toBeCloseTo(Math.PI);
  });

  it('copies a 1x2 source to a 1x2 dest', () => {
    const source = new Float32Array([1, 2]);
    const dest = new Float32Array([0, 0]);
    const result = uploadCopyDownload(
        source, [1, 2], [0, 0], [1, 2], [0, 0], [1, 2], dest, [1, 2]);
    expect(result.length).toEqual(2);
    expect(result[0]).toEqual(1);
    expect(result[1]).toEqual(2);
  });

  it('copies a 2x1 source to a 2x1 dest', () => {
    const source = new Float32Array([1, 2]);
    const dest = new Float32Array([0, 0]);
    const result = uploadCopyDownload(
        source, [2, 1], [0, 0], [2, 1], [0, 0], [2, 1], dest, [2, 1]);
    expect(result.length).toEqual(2);
    expect(result[0]).toEqual(1);
    expect(result[1]).toEqual(2);
  });

  it('copies a 2x2 source to a 2x2 dest', () => {
    const source = new Float32Array([1, 2, 3, 4]);
    const dest = new Float32Array([0, 0, 0, 0]);
    const result = uploadCopyDownload(
        source, [2, 2], [0, 0], [2, 2], [0, 0], [2, 2], dest, [2, 2]);
    expect(result.length).toEqual(4);
    expect(result[0]).toEqual(1);
    expect(result[1]).toEqual(2);
    expect(result[2]).toEqual(3);
    expect(result[3]).toEqual(4);
  });

  it('copies inner 2x2 from a 4x4 source to a 2x2 dest', () => {
    const source = new Float32Array(16);
    source[5] = 10;
    source[6] = 11;
    source[9] = 12;
    source[10] = 13;
    const dest = new Float32Array(4);
    const result = uploadCopyDownload(
        source, [4, 4], [1, 1], [2, 2], [0, 0], [2, 2], dest, [2, 2]);
    expect(result.length).toEqual(4);
    expect(result[0]).toEqual(10);
    expect(result[1]).toEqual(11);
    expect(result[2]).toEqual(12);
    expect(result[3]).toEqual(13);
  });

  it('copies a 1x4 row from source into a 2x2 dest', () => {
    const source = new Float32Array([1, 2, 3, 4]);
    const dest = new Float32Array(4);
    const result = uploadCopyDownload(
        source, [1, 4], [0, 0], [1, 4], [0, 0], [2, 2], dest, [2, 2]);
    expect(result.length).toEqual(4);
    expect(result[0]).toEqual(1);
    expect(result[1]).toEqual(2);
    expect(result[2]).toEqual(3);
    expect(result[3]).toEqual(4);
  });

  it('copies a 1x4 row from source into a 4x1 dest', () => {
    const source = new Float32Array([1, 2, 3, 4]);
    const dest = new Float32Array(4);
    const result = uploadCopyDownload(
        source, [1, 4], [0, 0], [1, 4], [0, 0], [4, 1], dest, [4, 1]);
    expect(result.length).toEqual(4);
    expect(result[0]).toEqual(1);
    expect(result[1]).toEqual(2);
    expect(result[2]).toEqual(3);
    expect(result[3]).toEqual(4);
  });

  it('copies a column from source into a dest row vector', () => {
    const source = new Float32Array(10 * 10);
    for (let i = 0; i < 10; ++i) {
      source[3 + (i * 10)] = i + 1;
    }
    const dest = new Float32Array(10);
    const result = uploadCopyDownload(
        source, [10, 10], [0, 3], [10, 1], [0, 0], [1, 10], dest, [1, 10]);
    test_util.expectArraysClose(
        result, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
  });

  it('doesn\'t touch destination pixels outside of the source box', () => {
    const source = new Float32Array([1]);
    const dest = new Float32Array([Math.PI, 0]);
    const result = uploadCopyDownload(
        source, [1, 1], [0, 0], [1, 1], [0, 1], [1, 1], dest, [1, 2]);
    expect(result[0]).toBeCloseTo(Math.PI);
    expect(result[1]).toEqual(1);
  });

  it('accumulates results from previous copies into dest texture', () => {
    const shape: [number, number] = [10, 10];
    const size: [number, number] = [10, 1];
    const sourceVals = new Float32Array(100);
    for (let i = 0; i < 100; ++i) {
      sourceVals[i] = i;
    }

    const gpgpu = new GPGPUContext();
    const texManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, texManager);

    const program = new Copy2DProgram(size[1], size[1]);
    const source = Array2D.new(shape, sourceVals);
    const dest = Array2D.zeros(shape);

    const binary = gpgpu_math.compileProgram(gpgpu, program, [source], dest);

    for (let i = 0; i < 10; ++i) {
      const offset: [number, number] = [0, i];
      const customSetup = program.getCustomSetupFunc(offset, offset, size);
      gpgpu_math.runProgram(binary, [source], dest, customSetup);
    }
    const res = dest.getValues();

    source.dispose();
    dest.dispose();
    texManager.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();

    test_util.expectArraysClose(res, sourceVals);
  });
});
