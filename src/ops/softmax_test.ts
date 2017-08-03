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
import {NDArrayMathCPU} from '../math/math_cpu';
import {Array1D, Scalar} from '../math/ndarray';
import {TensorArrayMap} from '../tensor_array_map';

import {SoftmaxCrossEntropyCost} from './softmax';

describe('softmax cross entropy cost', () => {
  let math: NDArrayMathCPU;
  let logitsTensor: Tensor;
  let labelTensor: Tensor;
  let yTensor: Tensor;
  let activations: TensorArrayMap;
  let gradients: TensorArrayMap;

  beforeEach(() => {
    math = new NDArrayMathCPU();
    activations = new TensorArrayMap();
    gradients = new TensorArrayMap();
  });

  afterEach(() => {
    activations.disposeArray(logitsTensor);
    activations.disposeArray(yTensor);
    gradients.disposeArray(logitsTensor);
    gradients.disposeArray(yTensor);
  });

  it('matches theory', () => {
    // Verify that when having softmax + cross entropy,
    // dE/dx = y - t, which is the theoretical result.
    const logits = Array1D.new([1, 2, 3]);
    const label = Array1D.new([0.3, 0.6, 0.1]);
    const softmaxLogits = math.softmax(logits);

    logitsTensor = new Tensor(logits.shape);
    labelTensor = new Tensor(label.shape);
    yTensor = new Tensor([]);

    activations.set(logitsTensor, logits);
    activations.set(labelTensor, label);

    const op = new SoftmaxCrossEntropyCost(logitsTensor, labelTensor, yTensor);

    op.feedForward(math, activations);
    const y = activations.get(yTensor);

    expect(y.get(0)).toBeCloseTo(
        -Math.log(softmaxLogits.get(0)) * label.get(0) +
            -Math.log(softmaxLogits.get(1)) * label.get(1) +
            -Math.log(softmaxLogits.get(2)) * label.get(2),
        3);

    const dy = Scalar.new(1);
    gradients.set(yTensor, dy);

    op.backProp(math, activations, gradients);

    const dLogits = gradients.get(logitsTensor);
    expect(dLogits.get(0)).toBeCloseTo(softmaxLogits.get(0) - label.get(0), 6);
    expect(dLogits.get(1)).toBeCloseTo(softmaxLogits.get(1) - label.get(1), 6);
    expect(dLogits.get(2)).toBeCloseTo(softmaxLogits.get(2) - label.get(2), 6);
  });
});
