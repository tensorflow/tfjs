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

import {ENV} from '../../environment';
import * as conv_util from '../../math/conv_util';
import {Array3D} from '../../math/ndarray';
import * as test_util from '../../test_util';
import {Tensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';
import {MaxPool} from './max_pool';

describe('Max pool', () => {
  const math = ENV.math;
  let xTensor: Tensor;
  let yTensor: Tensor;
  let activations: TensorArrayMap;
  let gradients: SummedTensorArrayMap;

  beforeEach(() => {
    activations = new TensorArrayMap();
    gradients = new SummedTensorArrayMap(math);
  });

  afterEach(() => {
    activations.disposeArray(xTensor);
    activations.disposeArray(yTensor);
    gradients.disposeArray(xTensor);
    gradients.disposeArray(yTensor);
  });

  it('Simple MaxPool', () => {
    const fSize = 2;
    const stride = 1;
    const pad = 0;
    const depth = 1;

    const x = Array3D.new([3, 3, depth], [1, 2, 3, 4, 5, 6, 7, 9, 8]);

    xTensor = new Tensor(x.shape);
    yTensor = new Tensor(conv_util.computeOutputShape3D(
        x.shape, fSize, x.shape[2], stride, pad));

    activations.set(xTensor, x);

    const op = new MaxPool(xTensor, yTensor, fSize, stride, pad);

    op.feedForward(math, activations);

    // Feed forward.
    const y = activations.get(yTensor) as Array3D<'float32'>;
    const expectedResult = Array3D.new([2, 2, depth], [5, 6, 9, 9]);
    expect(expectedResult.equals(y)).toBe(true);

    // Backprop.
    const dy = Array3D.new([2, 2, depth], [50, 60, 90, 80]);
    gradients.add(yTensor, dy);

    op.backProp(math, activations, gradients);

    const dx = gradients.get(xTensor) as Array3D<'float32'>;
    const expectedBackprop =
        Array3D.new([3, 3, depth], [0, 0, 0, 0, 50, 60, 0, 170, 0]);
    expect(expectedBackprop.equals(dx)).toBe(true);
  });

  it('MaxPool depth = 2', () => {
    const fSize = 2;
    const stride = 2;
    const pad = 0;
    const depth = 2;

    const x = Array3D.new([4, 4, depth], [
      1, 11, 2,  22,  3,  33,  4,  44,  5,  55,  6,  66,  7,  77,  8,  88,
      9, 99, 10, 100, 11, 110, 12, 120, 13, 130, 14, 140, 15, 150, 16, 160
    ]);

    xTensor = new Tensor(x.shape);
    yTensor = new Tensor(conv_util.computeOutputShape3D(
        x.shape, fSize, x.shape[2], stride, pad));

    activations.set(xTensor, x);

    const op = new MaxPool(xTensor, yTensor, fSize, stride, pad);

    op.feedForward(math, activations);

    // Feed forward.
    const y = activations.get(yTensor);
    const expectedResult =
        Array3D.new([2, 2, 2], [6, 66, 8, 88, 14, 140, 16, 160]);
    test_util.expectArraysClose(y.dataSync(), expectedResult.dataSync());
  });

  it('MaxPool depth = 2, with some negative numbers', () => {
    const fSize = 2;
    const stride = 2;
    const pad = 0;

    const x = Array3D.new([4, 4, 2], [
      -1, 11, 2,  22,  3,   33,  4,  44,  5,  55,  6,  -66, 7,  -77, 8,  88,
      9,  99, 10, 100, -11, 110, 12, 120, 13, 130, 14, 140, 15, 150, 16, -160
    ]);

    xTensor = new Tensor(x.shape);
    yTensor = new Tensor(conv_util.computeOutputShape3D(
        x.shape, fSize, x.shape[2], stride, pad));

    activations.set(xTensor, x);

    const op = new MaxPool(xTensor, yTensor, fSize, stride, pad);
    op.feedForward(math, activations);

    // Feed forward.
    const y = activations.get(yTensor);
    const expectedResult =
        Array3D.new([2, 2, 2], [6, 55, 8, 88, 14, 140, 16, 150]);

    test_util.expectArraysClose(y.dataSync(), expectedResult.dataSync());
  });

  it('MaxPool downsampling depth is preserved', () => {
    const fSize = 2;
    const stride = 2;
    const pad = 0;

    const x = Array3D.randNormal([6, 6, 5]);

    xTensor = new Tensor(x.shape);
    yTensor = new Tensor(conv_util.computeOutputShape3D(
        x.shape, fSize, x.shape[2], stride, pad));

    activations.set(xTensor, x);

    const op = new MaxPool(xTensor, yTensor, fSize, stride, pad);
    op.feedForward(math, activations);

    const y = activations.get(yTensor);
    expect(y.shape).toEqual([3, 3, 5]);
  });
});
