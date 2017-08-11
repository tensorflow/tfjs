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
import * as max_pool_backprop_gpu from './max_pool_backprop_gpu';
import * as max_pool_gpu from './max_pool_gpu';

describe('max_pool_backprop_gpu', () => {

  function uploadMaxPoolBackpropDownload(
      dy: Array3D, x: Array3D, fSize: number, origStride: number,
      origPad: number): Float32Array {
    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);

    const depth = dy.shape[2];
    const src = max_pool_backprop_gpu.getFragmentShaderMaxPoolBackprop(
        dy.shape, fSize, origStride, origPad);
    const program = gpgpu.createProgram(src);

    // Upload dy.
    const dyTexShapeRC = conv_util.computeTexShapeFrom3D(dy.shape);
    const dyTex = gpgpu.createMatrixTexture(dyTexShapeRC[0], dyTexShapeRC[1]);
    gpgpu.uploadMatrixToTexture(
        dyTex, dyTexShapeRC[0], dyTexShapeRC[1], dy.getValues());

    // Upload x.
    const xTexShapeRC = conv_util.computeTexShapeFrom3D(x.shape);
    const xTex = gpgpu.createMatrixTexture(xTexShapeRC[0], xTexShapeRC[1]);
    gpgpu.uploadMatrixToTexture(
        xTex, xTexShapeRC[0], xTexShapeRC[1], x.getValues());

    // Compute max positions.
    const maxPoolResultShape = conv_util.computeOutputShape3D(
        x.shape, fSize, x.shape[2], origStride, origPad);
    const maxPoolResultTexShape =
        conv_util.computeTexShapeFrom3D(maxPoolResultShape);
    const maxPoolPositionsResultTex = gpgpu.createMatrixTexture(
        maxPoolResultTexShape[0], maxPoolResultTexShape[1]);
    const maxPoolPositionsSrc =
        max_pool_gpu.getFragmentShaderMaxPoolPositionsSource(
            x.shape, fSize, origStride, origPad);
    const maxPoolPositionsProgram = gpgpu.createProgram(maxPoolPositionsSrc);
    max_pool_gpu.maxPoolCommon(
        gpgpu, maxPoolPositionsProgram, xTex, maxPoolPositionsResultTex,
        maxPoolResultTexShape);

    // Figure out the output shape by dilating the input.
    const dyRowsDilated = (dy.shape[0] - 1) * origStride + 1;
    const dyColsDilated = (dy.shape[1] - 1) * origStride + 1;
    const pad = fSize - 1 - origPad;
    const resultShapeRCD = conv_util.computeOutputShape3D(
        [dyRowsDilated, dyColsDilated, depth], fSize, depth, 1, pad);
    const resultTexRC = conv_util.computeTexShapeFrom3D(resultShapeRCD);
    const resultTex = gpgpu.createMatrixTexture(resultTexRC[0], resultTexRC[1]);
    max_pool_backprop_gpu.maxPoolBackprop(
        gpgpu, program, dyTex, maxPoolPositionsResultTex, resultTex,
        resultTexRC);
    const y = gpgpu.downloadMatrixFromTexture(
        resultTex, resultTexRC[0], resultTexRC[1]);

    gpgpu.deleteMatrixTexture(resultTex);
    gpgpu.deleteMatrixTexture(dyTex);
    gpgpu.deleteMatrixTexture(xTex);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();

    return y;
  }

  function compareToCPU(
      dyShape: [number, number, number], xShape: [number, number, number],
      fSize: number, origStride: number, origPad: number) {
    const dy = NDArray.randNormal<Array3D>(dyShape);
    const x = NDArray.randNormal<Array3D>(xShape);

    const mathCPU = new NDArrayMathCPU();
    const dxCPU = mathCPU.maxPoolBackprop(dy, x, fSize, origStride, origPad);
    const dxGPU =
        uploadMaxPoolBackpropDownload(dy, x, fSize, origStride, origPad);
    test_util.expectArraysClose(dxGPU, dxCPU.getValues(), 1e-5);
  }

  it('matches CPU on random input, d1=1,d2=1,f=2,s=1,p=0', () => {
    const depth = 1;
    const dyShape: [number, number, number] = [8, 8, depth];
    const xShape: [number, number, number] = [9, 9, depth];
    const fSize = 2;
    const stride = 1;
    const zeroPad = 0;
    compareToCPU(dyShape, xShape, fSize, stride, zeroPad);
  });

  it('matches CPU on random input, d=1,f=3,s=2,p=1', () => {
    const depth = 1;
    const dyShape: [number, number, number] = [7, 7, depth];
    const xShape: [number, number, number] = [13, 13, depth];
    const fSize = 3;
    const stride = 2;
    const zeroPad = 1;
    compareToCPU(dyShape, xShape, fSize, stride, zeroPad);
  });

  it('matches CPU on random input, d=4,f=2,s=1,p=0', () => {
    const depth = 4;
    const dyShape: [number, number, number] = [8, 8, depth];
    const xShape: [number, number, number] = [9, 9, depth];
    const fSize = 2;
    const stride = 1;
    const zeroPad = 0;
    compareToCPU(dyShape, xShape, fSize, stride, zeroPad);
  });

  it('matches CPU on random input, d=3,f=3,s=3,p=0', () => {
    const depth = 3;
    const dyShape: [number, number, number] = [7, 7, depth];
    const xShape: [number, number, number] = [21, 21, depth];
    const fSize = 3;
    const stride = 3;
    const zeroPad = 0;
    compareToCPU(dyShape, xShape, fSize, stride, zeroPad);
  });
});
