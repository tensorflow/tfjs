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

import * as conv_backprop_gpu from './conv_backprop_gpu';
import {GPGPUContext} from './gpgpu_context';

describe('conv_gpu derWeights', () => {

  function uploadDerWeightsDownload(
      x: Array3D, dy: Array3D, fSize: number, stride: number,
      zeroPad: number): Float32Array {
    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    const outputDepth = dy.shape[2];
    const src = conv_backprop_gpu.getFragmentShaderDerWeightsSource(
        x.shape, fSize, outputDepth, stride, zeroPad);
    const program = gpgpu.createProgram(src);
    const inputDepth = x.shape[2];

    // Upload x.
    const xTexShapeRC = conv_util.computeTexShapeFrom3D(x.shape);
    const xTex = gpgpu.createMatrixTexture(xTexShapeRC[0], xTexShapeRC[1]);
    gpgpu.uploadMatrixToTexture(
        xTex, xTexShapeRC[0], xTexShapeRC[1], x.getValues());

    // Upload dy.
    const dyTexShapeRC = conv_util.computeTexShapeFrom3D(dy.shape);
    const dyTex = gpgpu.createMatrixTexture(dyTexShapeRC[0], dyTexShapeRC[1]);
    gpgpu.uploadMatrixToTexture(
        dyTex, dyTexShapeRC[0], dyTexShapeRC[1], dy.getValues());

    const resultTexRC =
        conv_util.computeWeightsTexShape(inputDepth, outputDepth, fSize);
    const resultTex = gpgpu.createMatrixTexture(resultTexRC[0], resultTexRC[1]);
    conv_backprop_gpu.derWeights(
        gpgpu, program, xTex, dyTex, resultTex, resultTexRC);
    const dw = gpgpu.downloadMatrixFromTexture(
        resultTex, resultTexRC[0], resultTexRC[1]);

    gpgpu.deleteMatrixTexture(resultTex);
    gpgpu.deleteMatrixTexture(xTex);
    gpgpu.deleteMatrixTexture(dyTex);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();

    return dw;
  }

  function compareToCPU(
      inputShape: [number, number, number], fSize: number, outputDepth: number,
      stride: number, zeroPad: number) {
    const x = NDArray.randNormal<Array3D>(inputShape);
    const outputShape = conv_util.computeOutputShape3D(
        x.shape, fSize, outputDepth, stride, zeroPad);
    const dy = NDArray.randNormal<Array3D>(outputShape);

    const mathCPU = new NDArrayMathCPU();
    const dwCPU = mathCPU.conv2dDerWeights(x, dy, fSize, stride, zeroPad);

    const dwGPU = uploadDerWeightsDownload(x, dy, fSize, stride, zeroPad);
    test_util.expectArraysClose(dwGPU, dwCPU.getValues(), 1e-5);
  }

  it('matches CPU on random input, d1=3,d2=4,f=2,s=1,p=0', () => {
    const inputDepth = 3;
    const inputShape: [number, number, number] = [8, 8, inputDepth];
    const fSize = 2;
    const outputDepth = 4;
    const stride = 1;
    const zeroPad = 0;
    compareToCPU(inputShape, fSize, outputDepth, stride, zeroPad);
  });

  it('matches CPU on random input, d1=3,d2=4,f=3,s=1,p=1', () => {
    const inputDepth = 3;
    const inputShape: [number, number, number] = [8, 8, inputDepth];
    const fSize = 3;
    const outputDepth = 4;
    const stride = 1;
    const zeroPad = 1;
    compareToCPU(inputShape, fSize, outputDepth, stride, zeroPad);
  });

  it('matches CPU on random input, d1=3,d2=4,f=3,s=2,p=1', () => {
    const inputDepth = 3;
    const inputShape: [number, number, number] = [7, 7, inputDepth];
    const fSize = 3;
    const outputDepth = 4;
    const stride = 2;
    const zeroPad = 1;
    compareToCPU(inputShape, fSize, outputDepth, stride, zeroPad);
  });

  it('matches CPU on random input, d1=3,d2=4,f=3,s=3,p=1', () => {
    const inputDepth = 3;
    const inputShape: [number, number, number] = [7, 7, inputDepth];
    const fSize = 3;
    const outputDepth = 4;
    const stride = 3;
    const zeroPad = 1;
    compareToCPU(inputShape, fSize, outputDepth, stride, zeroPad);
  });
});
