/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ENV} from '../environment';
import {keep, tidy} from '../globals';
import {Node} from '../graph/graph';
import {SessionRuntime} from '../graph/session';
// tslint:disable-next-line:max-line-length
import {SummedTensorArrayMap, TensorArrayMap} from '../graph/tensor_array_map';
import {NDArrayMath} from '../math';
import {scalar, zerosLike} from '../ops/ops';
import {Scalar, Tensor, Variable} from '../tensor';
import {variable} from '../tensor';
import {NamedVariableMap} from '../types';

import {Optimizer} from './optimizer';

export class AdamOptimizer extends Optimizer {
  private c: Scalar;
  private eps: Scalar;
  private beta1: Scalar;
  private beta2: Scalar;
  private accBeta1: Variable;
  private accBeta2: Variable;
  private oneMinusBeta1: Scalar;
  private oneMinusBeta2: Scalar;
  private one: Scalar;

  private accumulatedFirstMoment: NamedVariableMap = {};
  private accumulatedSecondMoment: NamedVariableMap = {};

  constructor(
      protected learningRate: number, beta1: number, beta2: number,
      epsilon = 1e-8, specifiedVariableList?: Node[]) {
    super(learningRate, specifiedVariableList);
    this.c = keep(scalar(-learningRate));
    this.eps = keep(scalar(epsilon));
    // b1, b2 keep initial value of beta* hyperparameters.
    this.beta1 = keep(scalar(beta1));
    this.beta2 = keep(scalar(beta2));
    tidy(() => {
      // accB* will be updated by batch.
      this.accBeta1 = variable(scalar(beta1));
      this.accBeta2 = variable(scalar(beta2));
    });
    this.oneMinusBeta1 = keep(scalar(1 - beta1));
    this.oneMinusBeta2 = keep(scalar(1 - beta2));
    this.one = keep(scalar(1));
  }

  applyGradients(variableGradients: NamedVariableMap) {
    tidy(() => {
      const oneMinusAccBeta1 = this.one.sub(this.accBeta1);
      const oneMinusAccBeta2 = this.one.sub(this.accBeta2);

      for (const variableName in variableGradients) {
        const value = ENV.engine.registeredVariables[variableName];
        if (this.accumulatedFirstMoment[variableName] == null) {
          const trainable = false;
          this.accumulatedFirstMoment[variableName] =
              variable(zerosLike(value), trainable);
        }
        if (this.accumulatedSecondMoment[variableName] == null) {
          const trainable = false;
          this.accumulatedSecondMoment[variableName] =
              variable(zerosLike(value), trainable);
        }

        const gradient = variableGradients[variableName];
        const firstMoment = this.accumulatedFirstMoment[variableName];
        const secondMoment = this.accumulatedSecondMoment[variableName];

        const newFirstMoment =
            this.beta1.mul(firstMoment).add(this.oneMinusBeta1.mul(gradient));
        const newSecondMoment =
            this.beta2.mul(secondMoment)
                .add(this.oneMinusBeta2.mul(gradient.square()));

        const biasCorrectedFirstMoment = newFirstMoment.div(oneMinusAccBeta1);
        const biasCorrectedSecondMoment = newSecondMoment.div(oneMinusAccBeta2);

        this.accumulatedFirstMoment[variableName].assign(newFirstMoment);
        this.accumulatedSecondMoment[variableName].assign(newSecondMoment);

        const newValue = this.c
                             .mul(biasCorrectedFirstMoment.div(this.eps.add(
                                 biasCorrectedSecondMoment.sqrt())))
                             .add(value);
        value.assign(newValue);
      }

      this.accBeta1.assign(this.accBeta1.mul(this.beta1));
      this.accBeta2.assign(this.accBeta2.mul(this.beta2));
    });
  }

  beforeBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    super.beforeBatch(
        math, batchSize, runtime, activationArrayMap, gradientArrayMap);

    if (this.firstMomentGraph.size() === 0) {
      this.variableNodes.forEach(node => {
        this.firstMomentGraph.set(node.output, Tensor.zeros(node.output.shape));
      });
    }

    if (this.secondMomentGraph.size() === 0) {
      this.variableNodes.forEach(node => {
        this.secondMomentGraph.set(
            node.output, Tensor.zeros(node.output.shape));
      });
    }
  }

  afterBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    tidy(() => {
      const oneMinusAccBeta1 = this.one.sub(this.accBeta1);
      const oneMinusAccBeta2 = this.one.sub(this.accBeta2);

      this.variableNodes.forEach(node => {
        const oldVariable = activationArrayMap.get(node.output);
        const gradient = this.variableGradients.get(node.output);

        const oldFirstMoment = this.firstMomentGraph.get(node.output);
        const oldSecondMoment = this.secondMomentGraph.get(node.output);

        const newFirstMoment = math.scaledArrayAdd(
            this.beta1, oldFirstMoment, this.oneMinusBeta1, gradient);
        const newSecondMoment = math.scaledArrayAdd(
            this.beta2, oldSecondMoment, this.oneMinusBeta2, gradient.square());

        const biasCorrectedFirstMoment = newFirstMoment.div(oneMinusAccBeta1);
        const biasCorrectedSecondMoment = newSecondMoment.div(oneMinusAccBeta2);
        const variable = math.scaledArrayAdd(
            this.cGraph,
            biasCorrectedFirstMoment.div(
                this.eps.add(biasCorrectedSecondMoment.sqrt())),
            this.one, oldVariable);
        activationArrayMap.set(node.output, keep(variable));
        node.data = variable;

        this.firstMomentGraph.set(node.output, keep(newFirstMoment));
        this.secondMomentGraph.set(node.output, keep(newSecondMoment));

        oldVariable.dispose();
        gradient.dispose();
        oldFirstMoment.dispose();
        oldSecondMoment.dispose();
      });
      this.accBeta1.assign(this.accBeta1.mul(this.beta1));
      this.accBeta2.assign(this.accBeta2.mul(this.beta2));
    });

    this.variableGradients.dispose();
    this.variableGradients = new TensorArrayMap();
  }

  dispose() {
    super.dispose();
    this.c.dispose();
    this.eps.dispose();
    this.beta1.dispose();
    this.beta2.dispose();
    this.accBeta1.dispose();
    this.accBeta2.dispose();
    this.oneMinusBeta1.dispose();
    this.oneMinusBeta2.dispose();
    this.one.dispose();

    if (this.firstMomentGraph != null) {
      this.firstMomentGraph.dispose();
    }

    if (this.secondMomentGraph != null) {
      this.secondMomentGraph.dispose();
    }

    if (this.accumulatedFirstMoment != null) {
      Object.keys(this.accumulatedFirstMoment)
          .forEach(name => this.accumulatedFirstMoment[name].dispose());
    }

    if (this.accumulatedSecondMoment != null) {
      Object.keys(this.accumulatedSecondMoment)
          .forEach(name => this.accumulatedSecondMoment[name].dispose());
    }
  }

  // Average of gradient
  private firstMomentGraph = new TensorArrayMap();
  // Average of squared gradient
  private secondMomentGraph = new TensorArrayMap();
}
