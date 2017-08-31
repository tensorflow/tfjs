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

import {Tensor} from '../graph';
import * as conv_util from '../math/conv_util';
import {NDArrayMathCPU} from '../math/math_cpu';
import {Array1D, Array2D, Array3D, Array4D, NDArray} from '../math/ndarray';
import {TensorArrayMap, SummedTensorArrayMap} from '../tensor_array_map';

import {Convolution2D} from './convolution';

function assertNoNaNs(t: NDArray) {
  const values = t.getValues();
  for (let i = 0; i < values.length; ++i) {
    expect(isNaN(values[i])).toBe(false);
  }
}

describe('Convolution', () => {
  let math: NDArrayMathCPU;
  let wTensor: Tensor;
  let xTensor: Tensor;
  let bTensor: Tensor;
  let yTensor: Tensor;
  let activations: TensorArrayMap;
  let gradients: SummedTensorArrayMap;

  beforeEach(() => {
    math = new NDArrayMathCPU();
    activations = new TensorArrayMap();
    gradients = new SummedTensorArrayMap(math);
  });

  afterEach(() => {
    activations.disposeArray(wTensor);
    activations.disposeArray(xTensor);
    activations.disposeArray(bTensor);
    activations.disposeArray(yTensor);
    gradients.disposeArray(wTensor);
    gradients.disposeArray(xTensor);
    gradients.disposeArray(bTensor);
    gradients.disposeArray(yTensor);
  });

  it('Forward prop comparison with convnetjs', () => {
    const inputDepth = 3;
    const outputDepth = 2;
    const fieldSize = 3;
    const stride = 2;
    const zeroPad = 1;
    const weights2D =
        Array2D.new([fieldSize * fieldSize * inputDepth, outputDepth], [
          1,  -1, 1, 0, -1, 1, -1, 0,  -1, 0,  0, 1,  -1, 1, 1, 1,  1, 1,
          0,  1,  0, 0, 0,  1, -1, -1, 1,  0,  1, -1, 1,  1, 1, 1,  1, -1,
          -1, 0,  1, 0, 0,  0, 1,  -1, -1, -1, 1, 0,  -1, 1, 0, -1, 0, 1
        ]);

    const weights =
        weights2D.as4D(fieldSize, fieldSize, inputDepth, outputDepth);
    const biases = Array1D.new([1, 0]);
    const x2D = Array2D.new([25, inputDepth], [
      1, 2, 2, 0, 0, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 0,
      2, 2, 1, 1, 0, 0, 2, 1, 1, 0, 1, 2, 2, 0, 0, 2, 2, 1, 2,
      2, 2, 1, 2, 2, 2, 1, 1, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 1,
      0, 1, 2, 0, 0, 0, 0, 1, 0, 0, 2, 2, 1, 0, 2, 0, 0, 0
    ]);
    const x = x2D.as3D(5, 5, inputDepth);

    wTensor = new Tensor(weights.shape);
    xTensor = new Tensor(x.shape);
    bTensor = new Tensor(biases.shape);
    yTensor = new Tensor(conv_util.computeOutputShape3D(
        x.shape, fieldSize, outputDepth, stride, zeroPad));

    activations.set(wTensor, weights);
    activations.set(xTensor, x);
    activations.set(bTensor, biases);
    const conv = new Convolution2D(
        wTensor, xTensor, bTensor, yTensor, fieldSize, outputDepth, stride,
        zeroPad);
    conv.feedForward(math, activations);

    const result = activations.get(yTensor);

    expect(result.getValues()).toEqual(new Float32Array([
      7, -8, 8, -2, 7, -2, 5, 5, 4, 6, 1, 2, -1, 3, 7, -2, 1, 4
    ]));
  });

  it('Maintains the rows and cols of input', () => {
    const inputDepth = 3;
    const outputDepth = 2;
    const fSize = 3;
    const stride = 1;

    const weights =
        NDArray.randNormal<Array4D>([fSize, fSize, inputDepth, outputDepth]);
    const biases = NDArray.randNormal<Array1D>([outputDepth]);
    const x = NDArray.randNormal<Array3D>([5, 5, inputDepth]);

    wTensor = new Tensor(weights.shape);
    xTensor = new Tensor(x.shape);
    bTensor = new Tensor(biases.shape);
    yTensor = new Tensor(
        conv_util.computeOutputShape3D(x.shape, fSize, outputDepth, stride));

    activations.set(wTensor, weights);
    activations.set(xTensor, x);
    activations.set(bTensor, biases);

    const conv = new Convolution2D(
        wTensor, xTensor, bTensor, yTensor, fSize, outputDepth, stride);

    conv.feedForward(math, activations);

    const result = activations.get(yTensor);

    expect(result.shape).toEqual([5, 5, outputDepth]);
  });

  it('Can not maintain the rows and cols of input', () => {
    const inputDepth = 3;
    const outputDepth = 2;
    const fSize = 2;
    const stride = 1;

    const weights =
        NDArray.randNormal<Array4D>([fSize, fSize, inputDepth, outputDepth]);
    const biases = NDArray.randNormal<Array1D>([outputDepth]);
    const x = NDArray.randNormal<Array3D>([5, 5, inputDepth]);

    wTensor = new Tensor(weights.shape);
    xTensor = new Tensor(x.shape);
    bTensor = new Tensor(biases.shape);
    yTensor = new Tensor(
        conv_util.computeOutputShape3D(x.shape, fSize, outputDepth, stride));

    activations.set(wTensor, weights);
    activations.set(xTensor, x);
    activations.set(bTensor, biases);

    const conv = new Convolution2D(
        wTensor, xTensor, bTensor, yTensor, fSize, outputDepth, stride);

    conv.feedForward(math, activations);

    const result = activations.get(yTensor);

    expect(result.shape).toEqual([4, 4, outputDepth]);
  });

  it('Large convolution', () => {
    const inputDepth = 3;
    const fSize = 7;
    const outputDepth = 10;
    const stride = 1;
    const zeroPad = 1;

    const weights =
        NDArray.randNormal<Array4D>([fSize, fSize, inputDepth, outputDepth]);
    const biases = NDArray.randNormal<Array1D>([outputDepth]);
    const x = NDArray.randNormal<Array3D>([30, 30, inputDepth]);

    wTensor = new Tensor(weights.shape);
    xTensor = new Tensor(x.shape);
    bTensor = new Tensor(biases.shape);
    yTensor = new Tensor(conv_util.computeOutputShape3D(
        x.shape, fSize, outputDepth, stride, zeroPad));

    activations.set(wTensor, weights);
    activations.set(xTensor, x);
    activations.set(bTensor, biases);

    const conv = new Convolution2D(
        wTensor, xTensor, bTensor, yTensor, fSize, outputDepth, stride,
        zeroPad);

    conv.feedForward(math, activations);

    const result = activations.get(yTensor);

    assertNoNaNs(result);
    expect(result.shape).toEqual([26, 26, outputDepth]);
  });

  it('simple conv backprop with d1=d2=1 (input and output)', () => {
    // 3X3 image convolved with a 2x2 filter with no padding and stride 1.
    // To keep the test simple, we work with input and output depth of 1.
    const inputDepth = 1;
    const fSize = 2;
    const outputDepth = 1;
    const stride = 1;
    const zeroPad = 0;

    const x3d = NDArray.randNormal<Array3D>([3, 3, inputDepth]);
    const x = x3d.as2D(3, 3);
    const weights =
        NDArray.randNormal<Array4D>([fSize, fSize, inputDepth, outputDepth]);
    const biases = NDArray.randNormal<Array1D>([outputDepth]);

    wTensor = new Tensor(weights.shape);
    xTensor = new Tensor(x3d.shape);
    bTensor = new Tensor(biases.shape);
    yTensor = new Tensor(conv_util.computeOutputShape3D(
        x3d.shape, fSize, outputDepth, stride, zeroPad));

    activations.set(wTensor, weights);
    activations.set(xTensor, x3d);
    activations.set(bTensor, biases);
    const conv = new Convolution2D(
        wTensor, xTensor, bTensor, yTensor, fSize, outputDepth, stride,
        zeroPad);

    conv.feedForward(math, activations);

    const y = activations.get(yTensor);

    assertNoNaNs(y);

    expect(y.get(0, 0, 0))
        .toBeCloseTo(
            x.get(0, 0) * weights.get(0, 0, 0, 0) +
            x.get(0, 1) * weights.get(0, 1, 0, 0) +
            x.get(1, 0) * weights.get(1, 0, 0, 0) +
            x.get(1, 1) * weights.get(1, 1, 0, 0) + biases.get(0));

    expect(y.get(0, 1, 0))
        .toBeCloseTo(
            x.get(0, 1) * weights.get(0, 0, 0, 0) +
            x.get(0, 2) * weights.get(0, 1, 0, 0) +
            x.get(1, 1) * weights.get(1, 0, 0, 0) +
            x.get(1, 2) * weights.get(1, 1, 0, 0) + biases.get(0));

    expect(y.get(1, 0, 0))
        .toBeCloseTo(
            x.get(1, 0) * weights.get(0, 0, 0, 0) +
            x.get(1, 1) * weights.get(0, 1, 0, 0) +
            x.get(2, 0) * weights.get(1, 0, 0, 0) +
            x.get(2, 1) * weights.get(1, 1, 0, 0) + biases.get(0));

    expect(y.get(1, 1, 0))
        .toBeCloseTo(
            x.get(1, 1) * weights.get(0, 0, 0, 0) +
            x.get(1, 2) * weights.get(0, 1, 0, 0) +
            x.get(2, 1) * weights.get(1, 0, 0, 0) +
            x.get(2, 2) * weights.get(1, 1, 0, 0) + biases.get(0));

    const dy3d = NDArray.randNormal<Array3D>([2, 2, 1]);

    gradients.add(yTensor, dy3d);

    conv.backProp(math, activations, gradients);

    const dx3d = gradients.get(xTensor);

    // Since depth (last dim) is 1, we reduce indexing by converting 3D -> 2D.
    const dx = dx3d.as2D(3, 3);
    const dy = dy3d.as2D(2, 2);

    // Test dX.
    expect(dx.get(0, 0)).toBeCloseTo(dy.get(0, 0) * weights.get(0, 0, 0, 0));
    expect(dx.get(0, 1))
        .toBeCloseTo(
            dy.get(0, 0) * weights.get(0, 1, 0, 0) +
            dy.get(0, 1) * weights.get(0, 0, 0, 0));
    expect(dx.get(0, 2)).toBeCloseTo(dy.get(0, 1) * weights.get(0, 1, 0, 0));
    expect(dx.get(1, 1))
        .toBeCloseTo(
            dy.get(0, 0) * weights.get(1, 1, 0, 0) +
            dy.get(0, 1) * weights.get(1, 0, 0, 0) +
            dy.get(1, 0) * weights.get(0, 1, 0, 0) +
            dy.get(1, 1) * weights.get(0, 0, 0, 0));
    expect(dx.get(2, 1))
        .toBeCloseTo(
            dy.get(1, 0) * weights.get(1, 1, 0, 0) +
            dy.get(1, 1) * weights.get(1, 0, 0, 0));


    // Test dW.
    const dw = gradients.get(wTensor);

    expect(dw.get(0, 0, 0, 0))
        .toBeCloseTo(
            dy.get(0, 0) * x.get(0, 0) + dy.get(0, 1) * x.get(0, 1) +
            dy.get(1, 0) * x.get(1, 0) + dy.get(1, 1) * x.get(1, 1));
    expect(dw.get(1, 1, 0, 0))
        .toBeCloseTo(
            dy.get(0, 0) * x.get(1, 1) + dy.get(0, 1) * x.get(1, 2) +
            dy.get(1, 0) * x.get(2, 1) + dy.get(1, 1) * x.get(2, 2));

    // Test db (bias).
    const db = gradients.get(bTensor).get(0);

    expect(db).toBeCloseTo(
        dy.get(0, 0) + dy.get(0, 1) + dy.get(1, 0) + dy.get(1, 1));
  });

  it('conv backprop with d1=3 d2=7', () => {
    const fSize = 5;
    const inputDepth = 3;
    const outputDepth = 7;
    const stride = 1;
    const zeroPad = 1;

    const weights =
        NDArray.randNormal<Array4D>([fSize, fSize, inputDepth, outputDepth]);
    const biases = NDArray.randNormal<Array1D>([outputDepth]);
    const x = NDArray.randNormal<Array3D>([10, 10, inputDepth]);

    wTensor = new Tensor(weights.shape);
    xTensor = new Tensor(x.shape);
    bTensor = new Tensor(biases.shape);
    yTensor = new Tensor(conv_util.computeOutputShape3D(
        x.shape, fSize, outputDepth, stride, zeroPad));

    activations.set(wTensor, weights);
    activations.set(xTensor, x);
    activations.set(bTensor, biases);

    const conv = new Convolution2D(
        wTensor, xTensor, bTensor, yTensor, fSize, outputDepth, stride,
        zeroPad);

    conv.feedForward(math, activations);

    const result = activations.get(yTensor);

    assertNoNaNs(result);

    const dy = NDArray.randNormal<Array3D>(result.shape);

    gradients.add(yTensor, dy);

    conv.backProp(math, activations, gradients);

    const dx = gradients.get(xTensor);
    assertNoNaNs(dx);
  });
});
