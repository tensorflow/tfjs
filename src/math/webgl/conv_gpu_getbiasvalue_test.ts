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

import * as conv_gpu from './conv_gpu';
import {GPGPUContext} from './gpgpu_context';

describe('conv_gpu getBiasValue', () => {
  function createGetBiasValueProgram(
      gpgpu: GPGPUContext, outputDepth: number): WebGLProgram {
    const prologue = conv_gpu.getFragmentShaderPrologueSource();
    const uniforms = 'uniform float biasC;';
    const getBiasValue =
        conv_gpu.getFragmentShaderGetBiasValueSource(outputDepth);
    const main = `
      void main() {
        gl_FragColor = vec4(getBiasValue(biases, biasC), 0, 0, 0);
      }`;

    const src = [prologue, uniforms, getBiasValue, main].join('\n');
    return gpgpu.createProgram(src);
  }

  function uploadGetBiasValueDownload(
      biases: Float32Array, biasCol: number, outputDepth: number): number {
    const gpgpu = new GPGPUContext();
    const program = createGetBiasValueProgram(gpgpu, outputDepth);
    const biasesTex = gpgpu.createMatrixTexture(1, outputDepth);
    const resultTex = gpgpu.createMatrixTexture(1, 1);
    gpgpu.uploadMatrixToTexture(biasesTex, 1, outputDepth, biases);
    gpgpu.setOutputMatrixTexture(resultTex, 1, 1);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(biasesTex, 'biases', 2);
    gpgpu.gl.uniform1f(gpgpu.getUniformLocation('biasC'), biasCol);
    gpgpu.executeProgram();
    const result = gpgpu.downloadMatrixFromTexture(resultTex, 1, 1)[0];
    gpgpu.deleteMatrixTexture(resultTex);
    gpgpu.deleteMatrixTexture(biasesTex);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result;
  }

  it('returns the only bias value if output depth is 1', () => {
    const biases = new Float32Array([4]);
    const result = uploadGetBiasValueDownload(biases, 0, 1);
    expect(result).toEqual(4);
  });

  it('returns the requested column if < output depth', () => {
    const biases = new Float32Array([1, 2, 3, 4, 5]);
    const result =
        uploadGetBiasValueDownload(biases, biases.length - 1, biases.length);
    expect(result).toEqual(5);
  });

  it('wraps around to column 0 if column == output depth', () => {
    const biases = new Float32Array([6, 0, 0]);
    const result = uploadGetBiasValueDownload(biases, 3, 3);
    expect(result).toEqual(6);
  });

  it('wraps around twice if column == 2*output depth', () => {
    const biases = new Float32Array([7, 0, 0]);
    const result = uploadGetBiasValueDownload(biases, 6, 3);
    expect(result).toEqual(7);
  });

  it('selects value from column mod(biasC, outputDepth)', () => {
    const biases = new Float32Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    const result = uploadGetBiasValueDownload(biases, 2017, biases.length);
    expect(result).toEqual(biases[2017 % biases.length]);
  });
});
