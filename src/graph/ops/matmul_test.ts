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
import {Array1D, Array2D} from '../../math/ndarray';
import {Tensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

import {MatMul} from './matmul';

describe('add operation', () => {
  const math = ENV.math;

  let t1: Tensor;
  let t2: Tensor;
  let y: Tensor;
  let matmulOp: MatMul;
  let activations: TensorArrayMap;
  let gradients: SummedTensorArrayMap;

  beforeEach(() => {
    activations = new TensorArrayMap();
    gradients = new SummedTensorArrayMap(math);
  });

  afterEach(() => {
    activations.disposeArray(t1);
    activations.disposeArray(t2);
    activations.disposeArray(y);
    gradients.disposeArray(t1);
    gradients.disposeArray(t2);
    gradients.disposeArray(y);
  });

  it('matmul two NDArray2Ds', () => {
    const x1 = Array2D.new([2, 3], [1, 2, 3, 10, 20, 30]);
    const x2 = Array2D.new([3, 2], [2, 3, 4, 1, 2, 3]);

    t1 = new Tensor(x1.shape);
    t2 = new Tensor(x2.shape);
    y = new Tensor([x1.shape[0], x2.shape[1]]);

    activations.set(t1, x1);
    activations.set(t2, x2);

    matmulOp = new MatMul(t1, t2, y);
    matmulOp.feedForward(math, activations);
    const yVal = activations.get(y) as Array2D;

    expect(yVal.shape).toEqual([x1.shape[0], x2.shape[1]]);
    expect(yVal.get(0, 0))
        .toEqual(
            x1.get(0, 0) * x2.get(0, 0) + x1.get(0, 1) * x2.get(1, 0) +
            x1.get(0, 2) * x2.get(2, 0));
    expect(yVal.get(0, 1))
        .toEqual(
            x1.get(0, 0) * x2.get(0, 1) + x1.get(0, 1) * x2.get(1, 1) +
            x1.get(0, 2) * x2.get(2, 1));
    expect(yVal.get(1, 0))
        .toEqual(
            x1.get(1, 0) * x2.get(0, 0) + x1.get(1, 1) * x2.get(1, 0) +
            x1.get(1, 2) * x2.get(2, 0));
    expect(yVal.get(1, 1))
        .toEqual(
            x1.get(1, 0) * x2.get(0, 1) + x1.get(1, 1) * x2.get(1, 1) +
            x1.get(1, 2) * x2.get(2, 1));

    const dy = Array2D.new([2, 2], [1, 2, 3, 4]);
    gradients.add(y, dy);

    matmulOp.backProp(math, activations, gradients);

    const dx1 = gradients.get(t1) as Array2D;

    // dx1 = dy * x2T
    expect(dx1.shape).toEqual(x1.shape);
    expect(dx1.get(0, 0))
        .toEqual(dy.get(0, 0) * x2.get(0, 0) + dy.get(0, 1) * x2.get(0, 1));
    expect(dx1.get(0, 1))
        .toEqual(dy.get(0, 0) * x2.get(1, 0) + dy.get(0, 1) * x2.get(1, 1));
    expect(dx1.get(0, 2))
        .toEqual(dy.get(0, 0) * x2.get(2, 0) + dy.get(0, 1) * x2.get(2, 1));
    expect(dx1.get(1, 0))
        .toEqual(dy.get(1, 0) * x2.get(0, 0) + dy.get(1, 1) * x2.get(0, 1));
    expect(dx1.get(1, 1))
        .toEqual(dy.get(1, 0) * x2.get(1, 0) + dy.get(1, 1) * x2.get(1, 1));
    expect(dx1.get(1, 2))
        .toEqual(dy.get(1, 0) * x2.get(2, 0) + dy.get(1, 1) * x2.get(2, 1));

    const dx2 = gradients.get(t2) as Array2D;

    // dx2 = x1T * dy
    expect(dx2.shape).toEqual(x2.shape);
    expect(dx2.get(0, 0))
        .toEqual(x1.get(0, 0) * dy.get(0, 0) + x1.get(1, 0) * dy.get(1, 0));
    expect(dx2.get(0, 1))
        .toEqual(x1.get(0, 0) * dy.get(0, 1) + x1.get(1, 0) * dy.get(1, 1));
    expect(dx2.get(1, 0))
        .toEqual(x1.get(0, 1) * dy.get(0, 0) + x1.get(1, 1) * dy.get(1, 0));
    expect(dx2.get(1, 1))
        .toEqual(x1.get(0, 1) * dy.get(0, 1) + x1.get(1, 1) * dy.get(1, 1));
    expect(dx2.get(2, 0))
        .toEqual(x1.get(0, 2) * dy.get(0, 0) + x1.get(1, 2) * dy.get(1, 0));
    expect(dx2.get(2, 1))
        .toEqual(x1.get(0, 2) * dy.get(0, 1) + x1.get(1, 2) * dy.get(1, 1));
  });

  it('matrix times vector', () => {
    const inputSize = 3;
    const outputSize = 2;
    const x1 = Array2D.new([outputSize, inputSize], [1, 2, 0, 4, 3, 2]);
    const x2 = Array1D.new([1, 2, 3]);

    t1 = new Tensor(x1.shape);
    t2 = new Tensor(x2.shape);
    y = new Tensor([x1.shape[0], x2.shape[1]]);

    activations.set(t1, x1);
    activations.set(t2, x2);

    const op = new MatMul(t1, t2, y);
    op.feedForward(math, activations);

    const yVal = activations.get(y);
    expect(yVal.get(0)).toBe(5);
    expect(yVal.get(1)).toBe(16);

    // Back prop.
    const dy = Array1D.new([2, 3]);
    gradients.add(y, dy);

    op.backProp(math, activations, gradients);

    const dx1 = gradients.get(t1).as2D(x1.shape[0], x1.shape[1]);
    expect(dx1.get(0, 0)).toBe(dy.get(0) * x2.get(0));
    expect(dx1.get(0, 1)).toBe(dy.get(0) * x2.get(1));
    expect(dx1.get(0, 2)).toBe(dy.get(0) * x2.get(2));
    expect(dx1.get(1, 0)).toBe(dy.get(1) * x2.get(0));
    expect(dx1.get(1, 1)).toBe(dy.get(1) * x2.get(1));
    expect(dx1.get(1, 2)).toBe(dy.get(1) * x2.get(2));

    const dx2 = gradients.get(t2).as1D();
    expect(dx2.get(0))
        .toBe(x1.get(0, 0) * dy.get(0) + x1.get(1, 0) * dy.get(1));
    expect(dx2.get(1))
        .toBe(x1.get(0, 1) * dy.get(0) + x1.get(1, 1) * dy.get(1));
    expect(dx2.get(2))
        .toBe(x1.get(0, 2) * dy.get(0) + x1.get(1, 2) * dy.get(1));
  });

  it('vector times matrix', () => {
    const x1 = Array1D.new([1, 2, 3]);
    const x2 = Array2D.new([3, 2], [1, 2, 0, 4, 3, 2]);

    t1 = new Tensor(x1.shape);
    t2 = new Tensor(x2.shape);
    y = new Tensor([x1.shape[0], x2.shape[1]]);

    activations.set(t1, x1);
    activations.set(t2, x2);

    const op = new MatMul(t1, t2, y);
    op.feedForward(math, activations);

    const yVal = activations.get(y);
    expect(yVal.get(0)).toBe(10);
    expect(yVal.get(1)).toBe(16);

    // Back prop.
    const dy = Array1D.new([2, 3]);
    gradients.add(y, dy);

    op.backProp(math, activations, gradients);

    const dx1 = gradients.get(t1).as1D();
    expect(dx1.get(0))
        .toBe(dy.get(0) * x2.get(0, 0) + dy.get(1) * x2.get(0, 1));
    expect(dx1.get(1))
        .toBe(dy.get(0) * x2.get(1, 0) + dy.get(1) * x2.get(1, 1));
    expect(dx1.get(2))
        .toBe(dy.get(0) * x2.get(2, 0) + dy.get(1) * x2.get(2, 1));

    const dx2 = gradients.get(t2).as2D(x2.shape[0], x2.shape[1]);
    expect(dx2.get(0, 0)).toBe(x1.get(0) * dy.get(0));
    expect(dx2.get(0, 1)).toBe(x1.get(0) * dy.get(1));
    expect(dx2.get(1, 0)).toBe(x1.get(1) * dy.get(0));
    expect(dx2.get(1, 1)).toBe(x1.get(1) * dy.get(1));
    expect(dx2.get(2, 0)).toBe(x1.get(2) * dy.get(0));
    expect(dx2.get(2, 1)).toBe(x1.get(2) * dy.get(1));
  });
});
