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

export class AdamaxOptimizer extends Optimizer {
  constructor(
      protected learningRate: number, private beta1: number,
      private beta2: number, specifiedVariableList?: Node[]) {
    super(learningRate, specifiedVariableList);
    this.eps = Scalar.new(1e-8);
    // b1, b2 keep initial value of beta* hyperparameters.
    this.b1 = Scalar.new(this.beta1);
    this.b2 = Scalar.new(this.beta2);

    this.accB1 = Scalar.new(this.beta1);
  }

  applyGradients(variableGradients: NamedVariableMap) {
    throw new Error(`Adamax optimizer not yet implemented for eager mode.`);
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

    if (this.weightedInfNorm.size() === 0) {
      this.variableNodes.forEach(node => {
        this.weightedInfNorm.set(node.output, NDArray.zeros(node.output.shape));
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
        const oldWeightedInfNorm = this.weightedInfNorm.get(node.output);

        const newFirstMoment = math.scaledArrayAdd(
            this.b1, oldFirstMoment, math.subtract(this.one, this.b1),
            gradient);

        const ut0 = math.multiply(this.b2, oldWeightedInfNorm);
        const ut1 = math.abs(gradient);

        const newWeightedInfNorm = math.add(
            math.relu(math.subtract(ut0, ut1)),
            ut1);  // update with element-wise max

        const variable = math.scaledArrayAdd(
            this.one, oldVariable,
            math.divideStrict(this.cGraph, math.subtract(this.one, this.accB1)),
            math.divide(
                newFirstMoment, math.add(this.eps, newWeightedInfNorm)));

        activationArrayMap.set(node.output, keep(variable));
        node.data = variable;

        this.firstMoment.set(node.output, keep(newFirstMoment));
        this.weightedInfNorm.set(node.output, keep(newWeightedInfNorm));

        oldVariable.dispose();
        gradient.dispose();
        oldFirstMoment.dispose();
        oldWeightedInfNorm.dispose();
      });
      // Make sure to dispose old values.
      const oldAccB1 = this.accB1;
      this.accB1 = keep(math.multiply(this.accB1, this.b1));
      oldAccB1.dispose();
    });

    this.variableGradients.dispose();
    this.variableGradients = new TensorArrayMap();
  }

  dispose() {
    super.dispose();
    this.firstMoment.dispose();
    this.weightedInfNorm.dispose();
    this.eps.dispose();
    this.accB1.dispose();
    this.b1.dispose();
    this.b2.dispose();
  }

  // Average of 1st gradient
  private firstMoment = new TensorArrayMap();
  // Average of exponentially weighed infinity norm
  private weightedInfNorm = new TensorArrayMap();
  private eps: Scalar;
  private accB1: Scalar;
  private b1: Scalar;
  private b2: Scalar;
}
