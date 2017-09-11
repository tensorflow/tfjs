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
import {NDArrayMathCPU} from '../math_cpu';
import {Array1D, Array3D, initializeGPU, NDArray} from '../ndarray';

import {Conv2DDerBiasProgram} from './conv_backprop_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {TextureManager} from './texture_manager';

describe('conv_gpu derBias', () => {

  function uploadDerBiasDownload(dy: Array3D): Float32Array {
    const gpgpu = new GPGPUContext();
    const texManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, texManager);
    gpgpu.enableAutomaticDebugValidation(true);
    const program = new Conv2DDerBiasProgram(dy.shape);
    const out = Array1D.zeros([dy.shape[2]]);
    const binary = gpgpu_math.compileProgram(gpgpu, program, [dy], out);
    gpgpu_math.runProgram(binary, [dy], out);
    const result = out.getValues();

    texManager.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();

    return result;
  }

  function compareToCPU(dyShapeRCD: [number, number, number]) {
    const dy = NDArray.randNormal<Array3D>(dyShapeRCD);

    const mathCPU = new NDArrayMathCPU();
    const dBiasCPU = mathCPU.conv2dDerBias(dy);

    const dBiasGPU = uploadDerBiasDownload(dy);
    test_util.expectArraysClose(dBiasGPU, dBiasCPU.getValues(), 1e-5);
  }

  it('matches CPU on random input. dy shape [3, 3, 2]', () => {
    compareToCPU([3, 3, 2]);
  });

  it('matches CPU on random input. dy shape [5, 5, 1]', () => {
    compareToCPU([5, 5, 1]);
  });

  it('matches CPU on random input. dy shape [1, 1, 8]', () => {
    compareToCPU([1, 1, 8]);
  });
});
