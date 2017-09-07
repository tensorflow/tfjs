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
import {Array3D, initializeGPU, NDArray} from '../ndarray';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {MaxPool2DBackpropProgram} from './max_pool_backprop_gpu';
import {Pool2DProgram} from './pool_gpu';
import {TextureManager} from './texture_manager';

describe('max_pool_backprop_gpu', () => {

  function uploadMaxPoolBackpropDownload(
      dy: Array3D, x: Array3D, fSize: number, origStride: number,
      origPad: number): Float32Array {
    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    const textureManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, textureManager);

    const getPositions = true;
    const outDepth = x.shape[2];
    const convInfo = conv_util.computeConvInfo(
        x.shape, fSize, fSize, outDepth, origStride, origStride, origPad);
    const positionsProgram = new Pool2DProgram(convInfo, 'max', getPositions);
    const positionsRes = NDArray.zeros(positionsProgram.outputShape);
    const positionsBinary =
        gpgpu_math.compileProgram(gpgpu, positionsProgram, [x], positionsRes);
    gpgpu_math.runProgram(positionsBinary, [x], positionsRes);
    const program = new MaxPool2DBackpropProgram(convInfo);
    const res = NDArray.zeros(program.outputShape);
    const binary =
        gpgpu_math.compileProgram(gpgpu, program, [dy, positionsRes], res);
    gpgpu_math.runProgram(binary, [dy, positionsRes], res);

    const resValues = res.getValues();

    textureManager.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();
    return resValues;
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
