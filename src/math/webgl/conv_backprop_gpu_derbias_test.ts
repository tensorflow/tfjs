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

describe('conv_gpu derBias', () => {

  function uploadDerBiasDownload(dy: Array3D): Float32Array {
    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    const src = conv_backprop_gpu.getFragmentShaderDerBiasSource(dy.shape);
    const program = gpgpu.createProgram(src);

    // Upload dy.
    const dyTexShapeRC = conv_util.computeTexShapeFrom3D(dy.shape);
    const dyTex = gpgpu.createMatrixTexture(dyTexShapeRC[0], dyTexShapeRC[1]);
    gpgpu.uploadMatrixToTexture(
        dyTex, dyTexShapeRC[0], dyTexShapeRC[1], dy.getValues());

    const outputDepth = dy.shape[2];
    const resultTexRC = conv_util.computeBiasesTexShape(outputDepth);
    const resultTex = gpgpu.createMatrixTexture(resultTexRC[0], resultTexRC[1]);
    conv_backprop_gpu.derBias(gpgpu, program, dyTex, resultTex, resultTexRC);
    const db = gpgpu.downloadMatrixFromTexture(
        resultTex, resultTexRC[0], resultTexRC[1]);

    gpgpu.deleteMatrixTexture(resultTex);
    gpgpu.deleteMatrixTexture(dyTex);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();

    return db;
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
