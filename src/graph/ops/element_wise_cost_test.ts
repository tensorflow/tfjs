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
import {Array1D} from '../../math/ndarray';
import {Tensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';
import {MeanSquaredCost} from './element_wise_cost';

describe('MeanSquaredCost', () => {
  const math = ENV.math;

  let x1Tensor: Tensor;
  let x2Tensor: Tensor;
  let yTensor: Tensor;
  let meanSquaredCostOperation: MeanSquaredCost;
  let activations: TensorArrayMap;
  let gradients: SummedTensorArrayMap;

  beforeEach(() => {
    activations = new TensorArrayMap();
    gradients = new SummedTensorArrayMap(math);
  });

  afterEach(() => {
    activations.disposeArray(x1Tensor);
    activations.disposeArray(x2Tensor);
    activations.disposeArray(yTensor);
  });

  it('mean squared cost, forward & backward', () => {
    const x1 = Array1D.new([1, 2, 3, 4]);
    const x2 = Array1D.new([2, 4, 6, 8]);

    x1Tensor = new Tensor(x1.shape);
    x2Tensor = new Tensor(x2.shape);
    yTensor = new Tensor([]);

    activations.set(x1Tensor, x1);
    activations.set(x2Tensor, x2);

    meanSquaredCostOperation = new MeanSquaredCost(x1Tensor, x2Tensor, yTensor);
    meanSquaredCostOperation.feedForward(math, activations);
    meanSquaredCostOperation.backProp(math, activations, gradients);

    const y = activations.get(yTensor);
    expect(y.shape).toEqual([]);
    expect(y.dataSync()).toEqual(new Float32Array([30 / 8]));

    const dx1 = gradients.get(x1Tensor);
    const dx2 = gradients.get(x2Tensor);
    expect(dx1.shape).toEqual(x1.shape);
    expect(dx2.shape).toEqual(x2.shape);
    expect(dx1.dataSync()).toEqual(new Float32Array([-1, -2, -3, -4]));
    expect(dx2.dataSync()).toEqual(new Float32Array([1, 2, 3, 4]));
  });
});
