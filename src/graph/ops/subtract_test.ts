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
import {Array1D, Array2D, Scalar} from '../../math/ndarray';
import {Tensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';
import {Subtract} from './subtract';

describe('add operation', () => {
  const math = ENV.math;

  let t1: Tensor;
  let t2: Tensor;
  let y: Tensor;
  let subOp: Subtract;
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

  it('subtracts two 1-D tensors', () => {
    const x1 = Array1D.new([1, 2, 3]);
    const x2 = Array1D.new([3, 0, 3]);

    t1 = new Tensor(x1.shape);
    t2 = new Tensor(x2.shape);
    y = new Tensor(x1.shape);

    activations.set(t1, x1);
    activations.set(t2, x2);

    subOp = new Subtract(t1, t2, y);
    subOp.feedForward(math, activations);
    const yVal = activations.get(y);

    expect(yVal.shape).toEqual([3]);
    expect(yVal.dataSync()).toEqual(new Float32Array([-2, 2, 0]));

    const dy = Array1D.new([6, 7, 8]);
    gradients.add(y, dy);

    subOp.backProp(math, activations, gradients);

    const dx1 = gradients.get(t1);
    const dx2 = gradients.get(t2);

    expect(dx1.shape).toEqual(x1.shape);
    expect(dx1.dataSync()).toEqual(dy.dataSync());

    expect(dx2.shape).toEqual(x2.shape);
    expect(dx2.dataSync()).toEqual(new Float32Array([-6, -7, -8]));
  });

  it('subtracts two 2-D tensors', () => {
    const x1 = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const x2 = Array2D.new([2, 3], [9, 8, 7, 6, 5, 4]);

    t1 = new Tensor(x1.shape);
    t2 = new Tensor(x2.shape);
    y = new Tensor(x1.shape);

    activations.set(t1, x1);
    activations.set(t2, x2);

    subOp = new Subtract(t1, t2, y);
    subOp.feedForward(math, activations);
    const yVal = activations.get(y);

    expect(yVal.shape).toEqual([2, 3]);
    expect(yVal.dataSync()).toEqual(new Float32Array([-8, -6, -4, -2, 0, 2]));

    const dy = Array2D.new([2, 3], [10, 11, 12, 13, 14, 15]);
    gradients.add(y, dy);

    subOp.backProp(math, activations, gradients);

    const dx1 = gradients.get(t1);
    const dx2 = gradients.get(t2);

    expect(dx1.shape).toEqual(x1.shape);
    expect(dx1.dataSync()).toEqual(dy.dataSync());

    expect(dx2.shape).toEqual(x2.shape);
    expect(dx2.dataSync()).toEqual(new Float32Array([
      -10, -11, -12, -13, -14, -15
    ]));
  });

  it('ndarray - scalar', () => {
    const x1 = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const x2 = Scalar.new(2);

    t1 = new Tensor(x1.shape);
    t2 = new Tensor(x2.shape);
    y = new Tensor(x1.shape);

    activations.set(t1, x1);
    activations.set(t2, x2);

    subOp = new Subtract(t1, t2, y);
    subOp.feedForward(math, activations);
    const yVal = activations.get(y);

    expect(yVal.shape).toEqual([2, 3]);
    expect(yVal.dataSync()).toEqual(new Float32Array([-1, 0, 1, 2, 3, 4]));

    const dy = Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
    gradients.add(y, dy);

    subOp.backProp(math, activations, gradients);

    const dx1 = gradients.get(t1);
    const dx2 = gradients.get(t2);

    expect(dx1.shape).toEqual(x1.shape);
    expect(dx1.dataSync()).toEqual(dy.dataSync());

    expect(dx2.shape).toEqual(x2.shape);
    expect(dx2.get()).toEqual(-42);
  });

  it('scalar - ndarray', () => {
    const x1 = Scalar.new(2);
    const x2 = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);

    t1 = new Tensor(x1.shape);
    t2 = new Tensor(x2.shape);
    y = new Tensor(x1.shape);

    activations.set(t1, x1);
    activations.set(t2, x2);

    subOp = new Subtract(t1, t2, y);
    subOp.feedForward(math, activations);
    const yVal = activations.get(y);

    expect(yVal.shape).toEqual([2, 3]);
    expect(yVal.dataSync()).toEqual(new Float32Array([1, 0, -1, -2, -3, -4]));

    const dy = Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
    gradients.add(y, dy);

    subOp.backProp(math, activations, gradients);

    const dx1 = gradients.get(t1);
    const dx2 = gradients.get(t2);

    expect(dx1.shape).toEqual(x1.shape);
    expect(dx1.get()).toEqual(42);

    expect(dx2.shape).toEqual(x2.shape);
    expect(dx2.dataSync()).toEqual(new Float32Array([
      -2, -4, -6, -8, -10, -12
    ]));
  });

  it('throws when shapes of X1 and X2 do not match', () => {
    const x1 = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const x2 = Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);

    t1 = new Tensor(x1.shape);
    t2 = new Tensor(x2.shape);
    y = new Tensor(x1.shape);

    activations.set(t1, x1);
    activations.set(t2, x2);

    expect(() => new Subtract(t1, t2, y)).toThrowError();
  });
});
