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
import {Array1D, Array3D, Array4D, initializeGPU, NDArray} from '../ndarray';

import {Conv2DProgram} from './conv_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {TextureManager} from './texture_manager';

describe('conv_gpu', () => {

  function uploadConvolveDownload(
      xVals: Float32Array, xShape: [number, number, number],
      weights: Float32Array, biasVals: Float32Array|null, outDepth: number,
      filterSizes: [number, number]|number, strides: [number, number]|number,
      zeroPad?: number|'valid'|'same'): Float32Array {
    zeroPad = zeroPad != null ? zeroPad : 'same';

    const [filterHeight, filterWidth] = parseTuple(filterSizes);
    const [strideHeight, strideWidth] = parseTuple(strides);

    const x = Array3D.new(xShape, xVals);
    const wShape = conv_util.computeWeightsShape4D(
        xShape[2], outDepth, filterHeight, filterWidth);
    const W = Array4D.new(wShape, weights);
    const b = biasVals != null ? Array1D.new(biasVals) : null;

    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    const textureManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, textureManager);

    const convInfo = conv_util.computeConvInfo(
        xShape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth,
        zeroPad);
    const program = new Conv2DProgram(convInfo, biasVals != null);
    const res = NDArray.zeros(program.outputShape);
    const inputs = biasVals != null ? [x, W, b] : [x, W];
    const binary = gpgpu_math.compileProgram(gpgpu, program, inputs, res);
    gpgpu_math.runProgram(binary, inputs, res);
    const resValues = res.getValues();

    textureManager.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();
    return resValues;
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

  it('2x2x1 in, 1d out, 2x2 filter, s=2, bias=0, p=1', () => {
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

  it('2x2x1 in, 1d out, 2x1 filter, s=1, p=valid', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const w = new Float32Array([3, 5]);
    const bias: Float32Array = null;
    const result =
        uploadConvolveDownload(x, [2, 2, 1], w, bias, 1, [2, 1], 1, 'valid');
    expect(result).toEqual(new Float32Array([18, 26]));
  });

  it('2x2x1 in, 1d out, 1x2 filter, s=1, p=valid', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const w = new Float32Array([3, 5]);
    const bias: Float32Array = null;
    const result =
        uploadConvolveDownload(x, [2, 2, 1], w, bias, 1, [1, 2], 1, 'valid');
    expect(result).toEqual(new Float32Array([13, 29]));
  });

  it('2x2x1 in, 1d out, 2x2 filter, 1 stride, bias=-1', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const w = new Float32Array([3, 1, 5, 0]);
    const bias = new Float32Array([-1]);
    const result = uploadConvolveDownload(x, [2, 2, 1], w, bias, 1, 2, 1, 0);
    expect(result.length).toEqual(1);
    expect(result[0]).toBe(19);
  });

  it('2x2x1 in, 1d out, 2x2 filter, 1 stride, no bias', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const w = new Float32Array([3, 1, 5, 0]);
    const bias: Float32Array|null = null;
    const result = uploadConvolveDownload(x, [2, 2, 1], w, bias, 1, 2, 1, 0);
    expect(result.length).toEqual(1);
    expect(result[0]).toBe(20);
  });

  it('5x5x3 in, 2d out, 3x3 filter, s=2, p=1', () => {
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

function parseTuple(a: number|[number, number]): [number, number] {
  return typeof a === 'number' ? [a, a] : a;
}
