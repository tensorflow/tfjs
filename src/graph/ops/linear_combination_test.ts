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
import {Array1D, Scalar} from '../../math/ndarray';
import {Tensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

import {LinearCombination} from './linear_combination';

describe('Linear combination', () => {
  const math = ENV.math;
  let x1Tensor: Tensor;
  let x2Tensor: Tensor;
  let c1Tensor: Tensor;
  let c2Tensor: Tensor;
  let yTensor: Tensor;
  let activations: TensorArrayMap;
  let gradients: SummedTensorArrayMap;

  beforeEach(() => {
    activations = new TensorArrayMap();
    gradients = new SummedTensorArrayMap(math);
  });

  afterEach(() => {
    activations.disposeArray(x1Tensor);
    activations.disposeArray(x2Tensor);
    activations.disposeArray(c1Tensor);
    activations.disposeArray(c2Tensor);
    activations.disposeArray(yTensor);
    gradients.disposeArray(x1Tensor);
    gradients.disposeArray(x2Tensor);
    gradients.disposeArray(c1Tensor);
    gradients.disposeArray(c2Tensor);
    gradients.disposeArray(yTensor);
  });

  it('Simple linear combination', () => {
    const x1 = Array1D.new([1, 2, 3]);
    const x2 = Array1D.new([10, 20, 30]);
    const c1 = Scalar.new(3);
    const c2 = Scalar.new(2);

    x1Tensor = new Tensor(x1.shape);
    x2Tensor = new Tensor(x2.shape);
    c1Tensor = new Tensor(c1.shape);
    c2Tensor = new Tensor(c2.shape);
    yTensor = new Tensor([]);

    activations.set(x1Tensor, x1);
    activations.set(x2Tensor, x2);
    activations.set(c1Tensor, c1);
    activations.set(c2Tensor, c2);

    const op =
        new LinearCombination(x1Tensor, x2Tensor, c1Tensor, c2Tensor, yTensor);
    op.feedForward(math, activations);

    const y = activations.get(yTensor);
    expect(y.get(0)).toBe(x1.get(0) * c1.get() + x2.get(0) * c2.get());
    expect(y.get(1)).toBe(x1.get(1) * c1.get() + x2.get(1) * c2.get());
    expect(y.get(2)).toBe(x1.get(2) * c1.get() + x2.get(2) * c2.get());

    const dy = Array1D.new([2, 4, 6]);
    gradients.add(yTensor, dy);
    op.backProp(math, activations, gradients);

    const dx1 = gradients.get(x1Tensor);
    expect(dx1.get(0)).toBe(c1.get() * dy.get(0));
    expect(dx1.get(1)).toBe(c1.get() * dy.get(1));
    expect(dx1.get(2)).toBe(c1.get() * dy.get(2));

    const dx2 = gradients.get(x2Tensor);
    expect(dx2.get(0)).toBe(c2.get() * dy.get(0));
    expect(dx2.get(1)).toBe(c2.get() * dy.get(1));
    expect(dx2.get(2)).toBe(c2.get() * dy.get(2));

    const dc1 = gradients.get(c1Tensor);
    expect(dc1.get()).toBe(
        x1.get(0) * dy.get(0) + x1.get(1) * dy.get(1) + x1.get(2) * dy.get(2));

    const dc2 = gradients.get(c2Tensor);
    expect(dc2.get()).toBe(
        x2.get(0) * dy.get(0) + x2.get(1) * dy.get(1) + x2.get(2) * dy.get(2));
  });
});
