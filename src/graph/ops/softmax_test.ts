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
import * as test_util from '../../test_util';
import {Tensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

import {Softmax, SoftmaxCrossEntropyCost} from './softmax';

describe('softmax cross entropy cost', () => {
  const math = ENV.math;
  let logitsTensor: Tensor;
  let labelTensor: Tensor;
  let yTensor: Tensor;
  let activations: TensorArrayMap;
  let gradients: SummedTensorArrayMap;

  beforeEach(() => {
    activations = new TensorArrayMap();
    gradients = new SummedTensorArrayMap(math);
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

    test_util.expectNumbersClose(
        y.get(0),
        -Math.log(softmaxLogits.get(0)) * label.get(0) +
            -Math.log(softmaxLogits.get(1)) * label.get(1) +
            -Math.log(softmaxLogits.get(2)) * label.get(2));

    const dy = Scalar.new(1);
    gradients.add(yTensor, dy);

    op.backProp(math, activations, gradients);

    const dLogits = gradients.get(logitsTensor);
    test_util.expectNumbersClose(
        dLogits.get(0), softmaxLogits.get(0) - label.get(0));
    test_util.expectNumbersClose(
        dLogits.get(1), softmaxLogits.get(1) - label.get(1));
    test_util.expectNumbersClose(
        dLogits.get(2), softmaxLogits.get(2) - label.get(2));
  });
});

describe('softmax operation', () => {
  const math = ENV.math;
  let logitsTensor: Tensor;
  let yTensor: Tensor;
  let activations: TensorArrayMap;
  let gradients: SummedTensorArrayMap;

  beforeEach(() => {
    const math = ENV.math;
    activations = new TensorArrayMap();
    gradients = new SummedTensorArrayMap(math);
  });

  afterEach(() => {
    activations.disposeArray(logitsTensor);
    activations.disposeArray(yTensor);
    gradients.disposeArray(logitsTensor);
    gradients.disposeArray(yTensor);
  });

  it('matches theory', () => {
    const logits = Array1D.new([10, 0, -1]);
    const softmaxLogits = math.softmax(logits);

    logitsTensor = new Tensor(logits.shape);
    yTensor = new Tensor([]);

    activations.set(logitsTensor, logits);

    const op = new Softmax(logitsTensor, yTensor);

    op.feedForward(math, activations);
    const y = activations.get(yTensor);

    test_util.expectArraysClose(y.dataSync(), softmaxLogits.dataSync());

    const dy = Array1D.new([1, 10, 100]);
    gradients.add(yTensor, dy);

    op.backProp(math, activations, gradients);

    const dLogits = gradients.get(logitsTensor);
    const sumLogitsdLogits = math.sum(math.elementWiseMul(dy, softmaxLogits));

    test_util.expectNumbersClose(
        dLogits.get(0),
        (dy.get(0) - sumLogitsdLogits.get(0)) * softmaxLogits.get(0));
    test_util.expectNumbersClose(
        dLogits.get(1),
        (dy.get(1) - sumLogitsdLogits.get(0)) * softmaxLogits.get(1));
    test_util.expectNumbersClose(
        dLogits.get(2),
        (dy.get(2) - sumLogitsdLogits.get(0)) * softmaxLogits.get(2));
  });
});
