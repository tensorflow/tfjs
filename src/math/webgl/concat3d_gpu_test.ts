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
import {Array3D, initializeGPU, NDArray} from '../ndarray';
import {Concat3DProgram} from './concat3d_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {TextureManager} from './texture_manager';

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

  const program = new Concat3DProgram(aShape, bShape, axis);
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
