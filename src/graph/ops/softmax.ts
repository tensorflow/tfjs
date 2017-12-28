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
import {NDArrayMath} from '../../math/math';
import {Array1D, Scalar} from '../../math/ndarray';
import * as util from '../../util';
import {Tensor} from '../graph';
import * as graph_util from '../graph_util';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';
import {Operation} from './op';

export class Softmax extends Operation {
  constructor(private logitsTensor: Tensor, private output: Tensor) {
    super();
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const logits = inferenceArrays.get(this.logitsTensor) as Array1D;
    return math.scope((keep) => {
      inferenceArrays.set(this.output, keep(math.softmax(logits)));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    // grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax
    const y = inferenceArrays.get(this.output);
    const dy = gradientArrays.get(this.output);
    math.scope(() => {
      if (graph_util.shouldBackProp(this.logitsTensor)) {
        const dlogits = math.elementWiseMul(
            math.subtract(dy, math.sum(math.elementWiseMul(dy, y))), y);
        gradientArrays.add(this.logitsTensor, dlogits);
      }
    });
  }
}

export class SoftmaxCrossEntropyCost extends Operation {
  constructor(
      private logitsTensor: Tensor, private labelTensor: Tensor,
      private yTensor: Tensor) {
    super();
    this.softmaxTensor = new Tensor(logitsTensor.shape);
    this.epsilon = ENV.math.keep(Scalar.new(1e-5));
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const logits = inferenceArrays.get(this.logitsTensor) as Array1D;
    const label = inferenceArrays.get(this.labelTensor) as Array1D;

    math.scope((keep) => {
      const softmaxResult = math.softmax(logits);

      inferenceArrays.set(this.softmaxTensor, keep(softmaxResult));
      inferenceArrays.set(
          this.yTensor,
          keep(crossEntropyCost(math, softmaxResult, label, this.epsilon)));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    const softmax = inferenceArrays.get(this.softmaxTensor);
    const label = inferenceArrays.get(this.labelTensor);

    math.scope(() => {
      gradientArrays.add(this.logitsTensor, math.subtract(softmax, label));
    });
  }

  disposeTransientArrays(
      inferenceArrays: TensorArrayMap, gradientArrays: SummedTensorArrayMap) {
    inferenceArrays.disposeArray(this.softmaxTensor);
  }

  dispose() {
    this.epsilon.dispose();
  }

  private softmaxTensor: Tensor;
  private epsilon: Scalar;
}

export function crossEntropyCost(
    math: NDArrayMath, y: Array1D, target: Array1D,
    epsilon: Scalar): Scalar<'float32'> {
  util.assert(
      y.size === target.size, 'The output and target must be the same size');

  return math.scope(() => {
    const yPlusEps = math.scalarPlusArray(epsilon, y);
    const logOutput = math.log(yPlusEps);
    const tarLogOutput = math.elementWiseMul(target, logOutput);
    const costVector = math.neg(tarLogOutput);
    return math.sum(costVector);
  });
}
