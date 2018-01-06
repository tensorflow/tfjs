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

import {NDArrayMath} from '../../math/math';
import {NDArray, Scalar} from '../../math/ndarray';
import {Optimizer} from '../../math/optimizers/optimizer';
import {NamedVariableMap} from '../../util';
import {Node} from '../graph';
import {SessionRuntime} from '../session';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

export class AdamOptimizer extends Optimizer {
  constructor(
      protected learningRate: number, private beta1: number,
      private beta2: number, specifiedVariableList?: Node[]) {
    super(learningRate, specifiedVariableList);
    this.eps = Scalar.new(1e-8);
    // b1, b2 keep initial value of beta* hyperparameters.
    this.b1 = Scalar.new(this.beta1);
    this.b2 = Scalar.new(this.beta2);
    // accB* will be updated by batch.
    this.accB1 = Scalar.new(this.beta1);
    this.accB2 = Scalar.new(this.beta2);
  }

  applyGradients(variableGradients: NamedVariableMap) {
    throw new Error(`Adam optimizer not yet implemented for eager mode.`);
  }

  beforeBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    super.beforeBatch(
        math, batchSize, runtime, activationArrayMap, gradientArrayMap);

    if (this.firstMoment.size() === 0) {
      this.variableNodes.forEach(node => {
        this.firstMoment.set(node.output, NDArray.zeros(node.output.shape));
      });
    }

    if (this.secondMoment.size() === 0) {
      this.variableNodes.forEach(node => {
        this.secondMoment.set(node.output, NDArray.zeros(node.output.shape));
      });
    }
  }

  afterBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    math.scope((keep) => {
      this.variableNodes.forEach(node => {
        const oldVariable = activationArrayMap.get(node.output);
        const gradient = this.variableGradients.get(node.output);

        const oldFirstMoment = this.firstMoment.get(node.output);
        const oldSecondMoment = this.secondMoment.get(node.output);

        const newFirstMoment = math.scaledArrayAdd(
            this.b1, oldFirstMoment, math.subtract(this.one, this.b1),
            gradient);
        const gradientSquare = math.multiply(gradient, gradient);
        const newSecondMoment = math.scaledArrayAdd(
            this.b2, oldSecondMoment, math.subtract(this.one, this.b2),
            gradientSquare);

        const biasCorrectedFirstMoment =
            math.divide(newFirstMoment, math.subtract(this.one, this.accB1));
        const biasCorrectedSecondMoment =
            math.divide(newSecondMoment, math.subtract(this.one, this.accB2));

        const variable = math.scaledArrayAdd(
            this.cGraph,
            math.divide(
                biasCorrectedFirstMoment,
                math.add(math.sqrt(biasCorrectedSecondMoment), this.eps)),
            this.one, oldVariable);
        activationArrayMap.set(node.output, keep(variable));
        node.data = variable;

        this.firstMoment.set(node.output, keep(newFirstMoment));
        this.secondMoment.set(node.output, keep(newSecondMoment));

        oldVariable.dispose();
        gradient.dispose();
        oldFirstMoment.dispose();
        oldSecondMoment.dispose();
      });

      // Make sure to dispose old value objects.
      const oldAccB1 = this.accB1;
      const oldAccB2 = this.accB2;
      // accB* represents beta1 and beta2 to
      // the power t (the number of iteration).
      this.accB1 = keep(math.multiply(this.accB1, this.b1));
      this.accB2 = keep(math.multiply(this.accB2, this.b2));
      oldAccB1.dispose();
      oldAccB2.dispose();
    });

    this.variableGradients.dispose();
    this.variableGradients = new TensorArrayMap();
  }

  dispose() {
    super.dispose();
    this.firstMoment.dispose();
    this.secondMoment.dispose();
    this.eps.dispose();
    this.b1.dispose();
    this.b2.dispose();
    this.accB1.dispose();
    this.accB2.dispose();
  }

  // Average of gradient
  private firstMoment = new TensorArrayMap();
  // Average of squared gradient
  private secondMoment = new TensorArrayMap();
  private eps: Scalar<'float32'>;
  private b1: Scalar;
  private b2: Scalar;
  private accB1: Scalar;
  private accB2: Scalar;
}
