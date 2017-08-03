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

import * as conv_gpu from './conv_gpu';
import {GPGPUContext} from './gpgpu_context';

describe('conv_gpu', () => {

  function uploadConvolveDownload(
      x: Float32Array, aShapeRowColDepth: [number, number, number],
      weights: Float32Array, biases: Float32Array|null, resultDepth: number,
      fieldSize: number, stride: number, zeroPad?: number): Float32Array {
    zeroPad = zeroPad != null ?
        zeroPad :
        conv_util.computeDefaultPad(aShapeRowColDepth, fieldSize, stride);

    const xTexShapeRC: [number, number] =
        conv_util.computeTexShapeFrom3D(aShapeRowColDepth);

    const resultShapeRCD: [number, number, number] =
        conv_util.computeOutputShape3D(
            aShapeRowColDepth, fieldSize, resultDepth, stride, zeroPad);

    const weightsTexShapeRC: [number, number] =
        conv_util.computeWeightsTexShape(
            aShapeRowColDepth[2], resultDepth, fieldSize);

    const biasesTexShapeRC: [number, number] = [1, resultDepth];
    const resultTexShapeRC: [number, number] =
        conv_util.computeTexShapeFrom3D(resultShapeRCD);

    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);

    const shaderSource = conv_gpu.getFragmentShaderSource(
        aShapeRowColDepth, resultDepth, fieldSize, stride, zeroPad,
        biases != null);
    const program = gpgpu.createProgram(shaderSource);

    const xTex = gpgpu.createMatrixTexture(xTexShapeRC[0], xTexShapeRC[1]);
    const weightsTex =
        gpgpu.createMatrixTexture(weightsTexShapeRC[0], weightsTexShapeRC[1]);
    const biasesTex = biases != null ?
        gpgpu.createMatrixTexture(biasesTexShapeRC[0], biasesTexShapeRC[1]) :
        null;
    const resultTex =
        gpgpu.createMatrixTexture(resultTexShapeRC[0], resultTexShapeRC[1]);

    gpgpu.uploadMatrixToTexture(xTex, xTexShapeRC[0], xTexShapeRC[1], x);
    gpgpu.uploadMatrixToTexture(
        weightsTex, weightsTexShapeRC[0], weightsTexShapeRC[1], weights);

    if (biases != null) {
      gpgpu.uploadMatrixToTexture(
          biasesTex!, biasesTexShapeRC[0], biasesTexShapeRC[1], biases);
    }

    conv_gpu.convolve(
        gpgpu, program, xTex, weightsTex, biasesTex, resultTex,
        resultTexShapeRC);

    const result = gpgpu.downloadMatrixFromTexture(
        resultTex, resultTexShapeRC[0], resultTexShapeRC[1]);

    gpgpu.deleteMatrixTexture(resultTex);
    if (biasesTex != null) {
      gpgpu.deleteMatrixTexture(biasesTex);
    }
    gpgpu.deleteMatrixTexture(weightsTex);
    gpgpu.deleteMatrixTexture(xTex);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result;
  }

  function compareToCPU(
      xShape: [number, number, number], fSize: number, resultDepth: number,
      stride: number, pad: number) {
    const x = NDArray.randNormal<Array3D>(xShape);
    const weightsShape: [number, number, number, number] =
        [fSize, fSize, xShape[2], resultDepth];
    const weights = NDArray.randNormal<Array4D>(weightsShape);
    const biases = NDArray.randNormal<Array1D>([weightsShape[3]]);

    const mathCPU = new NDArrayMathCPU();
    const yCPU = mathCPU.conv2d(x, weights, biases, stride, pad);
    const yGPU = uploadConvolveDownload(
        x.getValues(), xShape, weights.getValues(), biases.getValues(),
        resultDepth, fSize, stride, pad);

    test_util.expectArraysClose(yGPU, yCPU.getValues(), 1e-5);
  }

  it('1x1x1 in, 1d out, 1x1 filter, 1 stride: [0] => [0]', () => {
    const a = new Float32Array([0]);
    const weights = new Float32Array([1]);
    const biases = new Float32Array([0]);
    const result =
        uploadConvolveDownload(a, [1, 1, 1], weights, biases, 1, 1, 1);
    expect(result).toBeCloseTo(0);
  });

  it('1x1x1 in, 1d out, 1x1 filter, 1 stride: [1] => [1]', () => {
    const a = new Float32Array([1]);
    const weights = new Float32Array([1]);
    const biases = new Float32Array([0]);
    const result =
        uploadConvolveDownload(a, [1, 1, 1], weights, biases, 1, 1, 1);
    expect(result).toBeCloseTo(1);
  });

  it('1x1x1 in, 1d out, 1x1 filter, 1 stride', () => {
    const a = new Float32Array([2]);
    const weights = new Float32Array([3]);
    const biases = new Float32Array([0]);
    const result =
        uploadConvolveDownload(a, [1, 1, 1], weights, biases, 1, 1, 1);
    expect(result).toBeCloseTo(6);
  });

  it('1x1x1 in, 1d out, 1x1 filter, 1 stride, null bias', () => {
    const a = new Float32Array([2]);
    const weights = new Float32Array([3]);
    const biases: Float32Array|null = null;
    const result =
        uploadConvolveDownload(a, [1, 1, 1], weights, biases, 1, 1, 1);
    expect(result).toBeCloseTo(6);
  });

  it('1x1x1 in, 1d out, 1x1 filter, 1 stride', () => {
    const a = new Float32Array([2]);
    const weights = new Float32Array([3]);
    const biases = new Float32Array([Math.PI]);
    const result =
        uploadConvolveDownload(a, [1, 1, 1], weights, biases, 1, 1, 1);
    expect(result).toBeCloseTo(6 + Math.PI);
  });

  it('1x1x2 in, 1d out, 1x1 filter, 1 stride', () => {
    const a = new Float32Array([1, 1]);
    const weights = new Float32Array([3, 5]);
    const biases = new Float32Array([0, 0]);
    const result =
        uploadConvolveDownload(a, [1, 1, 2], weights, biases, 1, 1, 1);
    expect(result).toBeCloseTo(8);
  });

  it('2x1x1 in, 1d out, 1x1 filter, 1 stride', () => {
    const a = new Float32Array([1, 2]);
    const weights = new Float32Array([5]);
    const biases = new Float32Array([0]);
    const result =
        uploadConvolveDownload(a, [2, 1, 1], weights, biases, 1, 1, 1);
    expect(result.length).toEqual(2);
    expect(result[0]).toBeCloseTo(5);
    expect(result[1]).toBeCloseTo(10);
  });

  it('2x1x1 in, 1d out, 1x1 filter, 1 stride', () => {
    const a = new Float32Array([1, 2]);
    const weights = new Float32Array([5]);
    const biases = new Float32Array([Math.PI]);
    const result =
        uploadConvolveDownload(a, [2, 1, 1], weights, biases, 1, 1, 1);
    expect(result.length).toEqual(2);
    expect(result[0]).toBeCloseTo(5 + Math.PI);
    expect(result[1]).toBeCloseTo(10 + Math.PI);
  });

  it('2x1x1 in, 2d out, 1x1 filter, 1 stride', () => {
    const a = new Float32Array([1, 2]);
    const weights = new Float32Array([5, 6]);
    const biases = new Float32Array([0, 0]);
    const result =
        uploadConvolveDownload(a, [2, 1, 1], weights, biases, 2, 1, 1);
    expect(result.length).toEqual(4);
    expect(result[0]).toBeCloseTo(a[0] * weights[0]);
    expect(result[1]).toBeCloseTo(a[0] * weights[1]);
    expect(result[2]).toBeCloseTo(a[1] * weights[0]);
    expect(result[3]).toBeCloseTo(a[1] * weights[1]);
  });

  it('2x1x1 in, 2d out, 1x1 filter, 1 stride', () => {
    const a = new Float32Array([1, 2]);
    const weights = new Float32Array([5, 6]);
    const biases = new Float32Array([100, 200]);
    const result =
        uploadConvolveDownload(a, [2, 1, 1], weights, biases, 2, 1, 1);
    expect(result.length).toEqual(4);
    expect(result[0]).toBeCloseTo((a[0] * weights[0]) + biases[0]);
    expect(result[1]).toBeCloseTo((a[0] * weights[1]) + biases[1]);
    expect(result[2]).toBeCloseTo((a[1] * weights[0]) + biases[0]);
    expect(result[3]).toBeCloseTo((a[1] * weights[1]) + biases[1]);
  });

  it('2x1x1 in, 3d out, 1x1 filter, 1 stride', () => {
    const a = new Float32Array([2, 4]);
    const weights = new Float32Array([3, 5, 7]);
    const biases = new Float32Array([0, 0, 0]);
    const result =
        uploadConvolveDownload(a, [2, 1, 1], weights, biases, 3, 1, 1, 0);
    expect(result.length).toEqual(2 * 3);
    expect(result[0]).toBeCloseTo(a[0] * weights[0]);
    expect(result[1]).toBeCloseTo(a[0] * weights[1]);
    expect(result[2]).toBeCloseTo(a[0] * weights[2]);
    expect(result[3]).toBeCloseTo(a[1] * weights[0]);
    expect(result[4]).toBeCloseTo(a[1] * weights[1]);
    expect(result[5]).toBeCloseTo(a[1] * weights[2]);
  });

  it('1x2x1 in, 1d out, 1x1 filter, 1 stride', () => {
    const a = new Float32Array([1, 2]);
    const weights = new Float32Array([5]);
    const biases = new Float32Array([0]);
    const result =
        uploadConvolveDownload(a, [1, 2, 1], weights, biases, 1, 1, 1);
    expect(result.length).toEqual(2);
    expect(result[0]).toBeCloseTo(5);
    expect(result[1]).toBeCloseTo(10);
  });

  it('2x1x2 in, 3d out, 1x1 filter, 1 stride', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const weights = new Float32Array([10, 11, 12, 13, 14, 15]);
    const biases = new Float32Array([0, 0, 0]);
    const result =
        uploadConvolveDownload(a, [2, 1, 2], weights, biases, 3, 1, 1);
    expect(result.length).toEqual(6);
    expect(result[0]).toBeCloseTo(a[0] * weights[0] + a[1] * weights[3]);
    expect(result[1]).toBeCloseTo(a[0] * weights[1] + a[1] * weights[4]);
    expect(result[2]).toBeCloseTo(a[0] * weights[2] + a[1] * weights[5]);
    expect(result[3]).toBeCloseTo(a[2] * weights[0] + a[3] * weights[3]);
    expect(result[4]).toBeCloseTo(a[2] * weights[1] + a[3] * weights[4]);
    expect(result[5]).toBeCloseTo(a[2] * weights[2] + a[3] * weights[5]);
  });

  it('2x2x1 in, 1d out, 2x2 filter, 1 stride', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const w = new Float32Array([3, 1, 5, 0]);
    const bias = new Float32Array([0]);
    const result = uploadConvolveDownload(x, [2, 2, 1], w, bias, 1, 2, 2, 1);
    expect(result.length).toEqual(4);
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(10);
    expect(result[2]).toBe(3);
    expect(result[3]).toBe(12);
  });

  it('2x2x1 in, 1d out, 2x2 filter, 1 stride', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const w = new Float32Array([3, 1, 5, 0]);
    const bias = new Float32Array([-1]);
    const result = uploadConvolveDownload(x, [2, 2, 1], w, bias, 1, 2, 1, 0);
    expect(result.length).toEqual(1);
    expect(result[0]).toBe(19);
  });

  it('2x2x1 in, 1d out, 2x2 filter, 1 stride, null bias', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const w = new Float32Array([3, 1, 5, 0]);
    const bias: Float32Array|null = null;
    const result = uploadConvolveDownload(x, [2, 2, 1], w, bias, 1, 2, 1, 0);
    expect(result.length).toEqual(1);
    expect(result[0]).toBe(20);
  });

  it('2x2x1 in, 1d out, 2x2 filter, 1 stride, zeropad = 1', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const w = new Float32Array([3, 1, 5, 0]);
    const bias = new Float32Array([0]);
    const result = uploadConvolveDownload(x, [2, 2, 1], w, bias, 1, 2, 2, 1);
    expect(result.length).toEqual(4);
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(10);
    expect(result[2]).toBe(3);
    expect(result[3]).toBe(12);
  });

  it('5x5x3 in, 2d out, 3x3 filter, 2 stride', () => {
    /*
      weights:       input:
        [ 1, -1,       [1, 2, 2, 0, 0, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2,
          1,  0,        1, 2, 2, 0, 2, 2, 1, 1, 0, 0, 2, 1, 1, 0, 1,
         -1,  1,        2, 2, 0, 0, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1,
         -1,  0,        1, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 1, 0, 1, 2,
         -1,  0,        0, 0, 0, 0, 1, 0, 0, 2, 2, 1, 0, 2, 0, 0, 0]
          0,  1,
         -1,  1,     biases:
          1,  1,       [1, 0]
          1,  1,
          0,  1,
          0,  0,
          0,  1,
         -1, -1,
          1,  0,
          1, -1,
          1,  1,
          1,  1,
          1, -1,
         -1,  0,
          1,  0,
          0,  0,
          1, -1,
         -1, -1,
          1,  0,
         -1,  1,
          0, -1,
          0,  1]
     */

    const input = new Float32Array([
      1, 2, 2, 0, 0, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 0,
      2, 2, 1, 1, 0, 0, 2, 1, 1, 0, 1, 2, 2, 0, 0, 2, 2, 1, 2,
      2, 2, 1, 2, 2, 2, 1, 1, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 1,
      0, 1, 2, 0, 0, 0, 0, 1, 0, 0, 2, 2, 1, 0, 2, 0, 0, 0
    ]);

    const weights = new Float32Array([
      1,  -1, 1, 0, -1, 1, -1, 0,  -1, 0,  0, 1,  -1, 1, 1, 1,  1, 1,
      0,  1,  0, 0, 0,  1, -1, -1, 1,  0,  1, -1, 1,  1, 1, 1,  1, -1,
      -1, 0,  1, 0, 0,  0, 1,  -1, -1, -1, 1, 0,  -1, 1, 0, -1, 0, 1
    ]);

    const biases = new Float32Array([1, 0]);

    const result =
        uploadConvolveDownload(input, [5, 5, 3], weights, biases, 2, 3, 2, 1);
    /*
      Filter centered at [0,0], zero-pad 1 column and 1 row
        0  0  0    0  0  0    0  0  0
        0  0  0    1  2  2    0  0  2
        0  0  0    1  2  2    0  2  2

      Weights, column [0]
        1  1 -1   -1 -1  0   -1  1  1
        0  0  0   -1  1  1    1  1  1
       -1  1  0    1 -1  1   -1  0  0

      Element-wise product (dot product before summation)
        0  0  0    0  0  0    0  0  0
        0  0  0   -1  2  2    0  0  2
        0  0  0    1 -2  2    0  0  0

      Sum of elements, plus bias of 1
        (-1 + 2 + 2 + 2 + 1 + -2 + 2) + 1 == 7
     */

    expect(result[0]).toBeCloseTo(7);

    test_util.expectArraysClose(
        result,
        new Float32Array(
            [7, -8, 8, -2, 7, -2, 5, 5, 4, 6, 1, 2, -1, 3, 7, -2, 1, 4]),
        0.00001);
  });

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
