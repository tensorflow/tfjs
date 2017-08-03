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
import {NDArrayMathCPU} from '../math_cpu';
import {Array3D, NDArray} from '../ndarray';

import {GPGPUContext} from './gpgpu_context';
import * as max_pool_gpu from './max_pool_gpu';

describe('max_pool_position', () => {
  function uploadMaxPoolPositionDownload(
      x: Float32Array, xShapeRowColDepth: [number, number, number],
      fieldSize: number, stride: number, pad: number): Float32Array {
    const xTexShapeRC: [number, number] =
        conv_util.computeTexShapeFrom3D(xShapeRowColDepth);

    const resultShapeRCD: [number, number, number] =
        conv_util.computeOutputShape3D(
            xShapeRowColDepth, fieldSize, xShapeRowColDepth[2], stride, pad);
    const resultTexShapeRC: [number, number] =
        conv_util.computeTexShapeFrom3D(resultShapeRCD);

    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);

    const shaderSource = max_pool_gpu.getFragmentShaderMaxPoolPositionsSource(
        xShapeRowColDepth, fieldSize, stride, pad);
    const program = gpgpu.createProgram(shaderSource);

    const xTex = gpgpu.createMatrixTexture(xTexShapeRC[0], xTexShapeRC[1]);
    const resultTex =
        gpgpu.createMatrixTexture(resultTexShapeRC[0], resultTexShapeRC[1]);

    gpgpu.uploadMatrixToTexture(xTex, xTexShapeRC[0], xTexShapeRC[1], x);

    max_pool_gpu.maxPoolCommon(
        gpgpu, program, xTex, resultTex, resultTexShapeRC);

    const result = gpgpu.downloadMatrixFromTexture(
        resultTex, resultTexShapeRC[0], resultTexShapeRC[1]);

    gpgpu.deleteMatrixTexture(resultTex);
    gpgpu.deleteMatrixTexture(xTex);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result;
  }

  function compareToCPU(
      xShape: [number, number, number], fSize: number, stride: number,
      pad: number) {
    const x = NDArray.randNormal<Array3D>(xShape);

    const mathCPU = new NDArrayMathCPU();
    const yCPU = mathCPU.maxPoolPositions(x, fSize, stride, pad);
    const yGPU = uploadMaxPoolPositionDownload(
        x.getValues(), x.shape, fSize, stride, pad);
    test_util.expectArraysClose(yGPU, yCPU.getValues(), 1e-5);
  }

  it('matches CPU on random input, d1=1,d2=1,f=2,s=1,p=0', () => {
    const depth = 1;
    const dyShape: [number, number, number] = [8, 8, depth];
    const fSize = 2;
    const stride = 1;
    const pad = 0;
    compareToCPU(dyShape, fSize, stride, pad);
  });

  it('matches CPU on random input, d=1,f=3,s=2,p=1', () => {
    const depth = 1;
    const inputShape: [number, number, number] = [7, 7, depth];
    const fSize = 3;
    const stride = 2;
    const pad = 1;
    compareToCPU(inputShape, fSize, stride, pad);
  });

  it('matches CPU on random input, d=4,f=2,s=1,p=0', () => {
    const depth = 4;
    const inputShape: [number, number, number] = [8, 8, depth];
    const fSize = 2;
    const stride = 1;
    const pad = 0;
    compareToCPU(inputShape, fSize, stride, pad);
  });

  it('matches CPU on random input, d=3,f=3,s=3,p=1', () => {
    const depth = 3;
    const inputShape: [number, number, number] = [7, 7, depth];
    const fSize = 3;
    const stride = 3;
    const pad = 1;
    compareToCPU(inputShape, fSize, stride, pad);
  });
});
