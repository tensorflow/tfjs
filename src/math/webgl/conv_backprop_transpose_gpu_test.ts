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
import {NDArrayMathCPU} from '../math_cpu';
import {Array1D, Array3D, Array4D, initializeGPU, NDArray} from '../ndarray';

import {Conv2DTransposeProgram} from './conv_backprop_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {TextureManager} from './texture_manager';

describe('conv_gpu transpose', () => {

  function uploadConvTransposeDownload(
      x: Array3D, W: Array4D, bias: Array1D|null, fSize: number,
      origStride: number, origPad: number): Float32Array {
    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    const textureManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, textureManager);
    const origInputDepth = W.shape[2];
    const program = new Conv2DTransposeProgram(
        x.shape, fSize, origInputDepth, origStride, origPad, bias != null);
    const res = NDArray.zeros(program.outputShape);
    const inputs = bias != null ? [x, W, bias] : [x, W];
    const binary = gpgpu_math.compileProgram(gpgpu, program, inputs, res);
    gpgpu_math.runProgram(binary, inputs, res);
    const resValues = res.getValues();

    textureManager.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();
    return resValues;
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
