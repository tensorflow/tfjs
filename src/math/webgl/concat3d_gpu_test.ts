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
import * as conv_util from '../conv_util';

import * as concat3d_gpu from './concat3d_gpu';
import {GPGPUContext} from './gpgpu_context';

describe('concat3d_gpu', () => {

  function uploadConcat3dDownload(
      x1: Float32Array, x2: Float32Array, x1ShapeRCD: [number, number, number],
      x2ShapeRCD: [number, number, number], axis: number): Float32Array {
    const x1TexShapeRC: [number, number] =
        conv_util.computeTexShapeFrom3D(x1ShapeRCD);
    const x2TexShapeRC: [number, number] =
        conv_util.computeTexShapeFrom3D(x2ShapeRCD);

    const resultShapeRCD = x1ShapeRCD.slice() as [number, number, number];
    resultShapeRCD[axis] += x2ShapeRCD[axis];
    const resultTexShapeRC = conv_util.computeTexShapeFrom3D(resultShapeRCD);

    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);

    const shaderSource = concat3d_gpu.getFragmentShaderSource(
        x1ShapeRCD, x2ShapeRCD, resultShapeRCD, axis);
    const program = gpgpu.createProgram(shaderSource);

    const x1Tex = gpgpu.createMatrixTexture(x1TexShapeRC[0], x1TexShapeRC[1]);
    const x2Tex = gpgpu.createMatrixTexture(x2TexShapeRC[0], x2TexShapeRC[1]);
    const resultTex =
        gpgpu.createMatrixTexture(resultTexShapeRC[0], resultTexShapeRC[1]);

    gpgpu.uploadMatrixToTexture(x1Tex, x1TexShapeRC[0], x1TexShapeRC[1], x1);
    gpgpu.uploadMatrixToTexture(x2Tex, x2TexShapeRC[0], x2TexShapeRC[1], x2);

    concat3d_gpu.concat3D(
        gpgpu, program, x1Tex, x2Tex, resultTex, resultTexShapeRC);

    const result = gpgpu.downloadMatrixFromTexture(
        resultTex, resultTexShapeRC[0], resultTexShapeRC[1]);

    gpgpu.deleteMatrixTexture(resultTex);
    gpgpu.deleteMatrixTexture(x1Tex);
    gpgpu.deleteMatrixTexture(x2Tex);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result;
  }

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
