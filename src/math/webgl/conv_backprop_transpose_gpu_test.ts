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
import {Array1D, Array3D, Array4D, NDArray} from '../ndarray';

import * as conv_backprop_gpu from './conv_backprop_gpu';
import {GPGPUContext} from './gpgpu_context';

describe('conv_gpu transpose', () => {

  function uploadConvTransposeDownload(
      x: Array3D, weights: Array4D, biases: Array1D|null, fSize: number,
      origStride: number, origPad: number): Float32Array {
    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    const origInputDepth = weights.shape[2];
    const origOutputDepth = weights.shape[3];
    const src = conv_backprop_gpu.getFragmentShaderConvTransposeSource(
        x.shape, fSize, origInputDepth, origStride, origPad, biases != null);
    const program = gpgpu.createProgram(src);

    // Upload x.
    const xTexShapeRC = conv_util.computeTexShapeFrom3D(x.shape);
    const xTex = gpgpu.createMatrixTexture(xTexShapeRC[0], xTexShapeRC[1]);
    gpgpu.uploadMatrixToTexture(
        xTex, xTexShapeRC[0], xTexShapeRC[1], x.getValues());

    // Upload weights.
    const wTexShapeRC = conv_util.computeWeightsTexShape(
        origInputDepth, origOutputDepth, fSize);
    const wTex = gpgpu.createMatrixTexture(wTexShapeRC[0], wTexShapeRC[1]);
    gpgpu.uploadMatrixToTexture(
        wTex, wTexShapeRC[0], wTexShapeRC[1], weights.getValues());

    const biasTexShapeRC = conv_util.computeBiasesTexShape(origInputDepth);
    const biasTex = biases != null ?
        gpgpu.createMatrixTexture(biasTexShapeRC[0], biasTexShapeRC[1]) :
        null;
    if (biasTex != null) {
      gpgpu.uploadMatrixToTexture(
          biasTex, biasTexShapeRC[0], biasTexShapeRC[1], biases!.getValues());
    }

    // Figure out the output shape by dilating the input.
    const xRowsDilated = (x.shape[0] - 1) * origStride + 1;
    const xColsDilated = (x.shape[1] - 1) * origStride + 1;
    const pad = fSize - 1 - origPad;
    const resultShapeRCD = conv_util.computeOutputShape3D(
        [xRowsDilated, xColsDilated, origOutputDepth], fSize, origInputDepth, 1,
        pad);
    const resultTexRC = conv_util.computeTexShapeFrom3D(resultShapeRCD);
    const resultTex = gpgpu.createMatrixTexture(resultTexRC[0], resultTexRC[1]);
    conv_backprop_gpu.convTranspose(
        gpgpu, program, xTex, wTex, biasTex, resultTex, resultTexRC);
    const y = gpgpu.downloadMatrixFromTexture(
        resultTex, resultTexRC[0], resultTexRC[1]);

    gpgpu.deleteMatrixTexture(resultTex);
    gpgpu.deleteMatrixTexture(xTex);
    gpgpu.deleteMatrixTexture(wTex);
    if (biasTex != null) {
      gpgpu.deleteMatrixTexture(biasTex);
    }
    gpgpu.deleteProgram(program);
    gpgpu.dispose();

    return y;
  }

  function compareToCPU(
      origInputShape: [number, number, number], fSize: number,
      origOutputDepth: number, origStride: number, origPad: number) {
    const [xNumRows, xNumCols, origInputDepth] = origInputShape;

    const x =
        NDArray.randNormal<Array3D>([xNumRows, xNumCols, origOutputDepth]);

    const weights = NDArray.randNormal<Array4D>(
        [fSize, fSize, origInputDepth, origOutputDepth]);
    const biases = NDArray.randNormal<Array1D>([origInputDepth]);

    const mathCPU = new NDArrayMathCPU();
    const yCPU =
        mathCPU.conv2dTranspose(x, weights, biases, origStride, origPad);
    const yGPU = uploadConvTransposeDownload(
        x, weights, biases, fSize, origStride, origPad);
    test_util.expectArraysClose(yGPU, yCPU.getValues(), 1e-5);
  }

  it('matches CPU on random input, d1=1,d2=1,f=2,s=1,p=0', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [8, 8, inputDepth];
    const fSize = 2;
    const outputDepth = 1;
    const stride = 1;
    const zeroPad = 0;
    compareToCPU(inputShape, fSize, outputDepth, stride, zeroPad);
  });

  it('matches CPU on random input, d1=1,d2=1,f=3,s=2,p=1', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [7, 7, inputDepth];
    const fSize = 3;
    const outputDepth = 1;
    const stride = 2;
    const zeroPad = 1;
    compareToCPU(inputShape, fSize, outputDepth, stride, zeroPad);
  });

  it('matches CPU on random input, d1=4,d2=3,f=2,s=1,p=0', () => {
    const inputDepth = 4;
    const inputShape: [number, number, number] = [8, 8, inputDepth];
    const fSize = 2;
    const outputDepth = 3;
    const stride = 1;
    const zeroPad = 0;
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
