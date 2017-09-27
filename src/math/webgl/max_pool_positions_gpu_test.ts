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
import * as conv_util from '../conv_util';
import {NDArrayMathCPU} from '../math_cpu';
import {Array3D, initializeGPU, NDArray} from '../ndarray';

import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {Pool2DProgram} from './pool_gpu';
import {TextureManager} from './texture_manager';

describe('max_pool_position', () => {
  function uploadMaxPoolPositionDownload(
      xVals: Float32Array, xShape: [number, number, number], fieldSize: number,
      stride: number, pad: number): Float32Array {
    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    const textureManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, textureManager);
    const getPositions = true;
    const outDepth = xShape[2];
    const convInfo = conv_util.computeConvInfo(
        xShape, fieldSize, fieldSize, outDepth, stride, stride, pad);
    const program = new Pool2DProgram(convInfo, 'max', getPositions);
    const res = NDArray.zeros(program.outputShape);
    const x = Array3D.new(xShape, xVals);
    const binary = gpgpu_math.compileProgram(gpgpu, program, [x], res);
    gpgpu_math.runProgram(binary, [x], res);
    const resValues = res.getValues();

    textureManager.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();
    return resValues;
  }

  function compareToCPU(
      xShape: [number, number, number], fSize: number, stride: number,
      pad: number) {
    const x = Array3D.randNormal(xShape);

    const mathCPU = new NDArrayMathCPU();
    const outDepth = x.shape[2];
    const convInfo = conv_util.computeConvInfo(
        x.shape, fSize, fSize, outDepth, stride, stride, pad);
    const yCPU = mathCPU.maxPoolPositions(x, convInfo);
    const yGPU = uploadMaxPoolPositionDownload(
        x.getValues(), x.shape, fSize, stride, pad);
    test_util.expectArraysClose(yGPU, yCPU.getValues());
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
