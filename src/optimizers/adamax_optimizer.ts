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
import {NamedVariableMap} from '../types';

import {Optimizer} from './optimizer';

export class AdamaxOptimizer extends Optimizer {
  private c: Scalar;
  private eps: Scalar;
  private accBeta1: Variable;
  private beta1: Scalar;
  private beta2: Scalar;
  private decay: Scalar;
  private oneMinusBeta1: Scalar;
  private one: Scalar;
  private iteration: Variable;

  private accumulatedFirstMoment: NamedVariableMap = {};
  private accumulatedWeightedInfNorm: NamedVariableMap = {};

  constructor(
      protected learningRate: number, beta1: number, beta2: number,
      epsilon = 1e-8, decay = 0.0,
      /** @deprecated */ specifiedVariableList?: Node[]) {
    super(learningRate, specifiedVariableList);
    this.c = keep(scalar(-learningRate));
    this.eps = keep(scalar(epsilon));
    // b1, b2 keep initial value of beta* hyperparameters.
    this.beta1 = keep(scalar(beta1));
    this.beta2 = keep(scalar(beta2));

    this.decay = keep(scalar(decay));

    tidy(() => {
      this.iteration = scalar(0).variable();
      this.accBeta1 = scalar(beta1).variable();
    });

    this.oneMinusBeta1 = keep(scalar(1 - beta1));
    this.one = keep(scalar(1));
  }

  applyGradients(variableGradients: NamedVariableMap) {
    tidy(() => {
      const oneMinusAccBeta1 = this.one.sub(this.accBeta1);
      const lr = this.c.div(this.one.add(this.decay.mul(this.iteration)));

      for (const variableName in variableGradients) {
        const value = ENV.engine.registeredVariables[variableName];
        if (this.accumulatedFirstMoment[variableName] == null) {
          const trainable = false;
          this.accumulatedFirstMoment[variableName] =
              zerosLike(value).variable(trainable);
        }
        if (this.accumulatedWeightedInfNorm[variableName] == null) {
          const trainable = false;
          this.accumulatedWeightedInfNorm[variableName] =
              zerosLike(value).variable(trainable);
        }

        const gradient = variableGradients[variableName];
        const firstMoment = this.accumulatedFirstMoment[variableName];
        const weightedInfNorm = this.accumulatedWeightedInfNorm[variableName];

        const newFirstMoment =
            this.beta1.mul(firstMoment).add(this.oneMinusBeta1.mul(gradient));

        const ut0 = this.beta2.mul(weightedInfNorm);
        const ut1 = gradient.abs();

        const newWeightedInfNorm = ut0.maximum(ut1);

        this.accumulatedFirstMoment[variableName].assign(newFirstMoment);
        this.accumulatedWeightedInfNorm[variableName].assign(
            newWeightedInfNorm);

        const newValue =
            lr.div(oneMinusAccBeta1)
                .mul(newFirstMoment.div(this.eps.add(newWeightedInfNorm)))
                .add(value);

        value.assign(newValue);
      }

      this.iteration.assign(this.iteration.add(this.one));
      this.accBeta1.assign(this.accBeta1.mul(this.beta1));
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

    if (this.weightedInfNormGraph.size() === 0) {
      this.variableNodes.forEach(node => {
        this.weightedInfNormGraph.set(
            node.output, Tensor.zeros(node.output.shape));
      });
    }
  }

  afterBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    tidy(() => {
      const lr = this.cGraph.div(this.one.add(this.decay.mul(this.iteration)));

      this.variableNodes.forEach(node => {
        const oldVariable = activationArrayMap.get(node.output);

        const gradient = this.variableGradients.get(node.output);
        const oldFirstMoment = this.firstMomentGraph.get(node.output);
        const oldWeightedInfNorm = this.weightedInfNormGraph.get(node.output);

        const newFirstMoment = math.scaledArrayAdd(
            this.beta1, oldFirstMoment, this.oneMinusBeta1, gradient);

        const ut0 = this.beta2.mul(oldWeightedInfNorm);
        const ut1 = gradient.abs();

        const newWeightedInfNorm = ut0.maximum(ut1);

        const variable = math.scaledArrayAdd(
            this.one, oldVariable, lr.div(this.one.sub(this.accBeta1)),
            newFirstMoment.div(this.eps.add(newWeightedInfNorm)));

        activationArrayMap.set(node.output, keep(variable));
        node.data = variable;

        this.firstMomentGraph.set(node.output, keep(newFirstMoment));
        this.weightedInfNormGraph.set(node.output, keep(newWeightedInfNorm));

        oldVariable.dispose();
        gradient.dispose();
        oldFirstMoment.dispose();
        oldWeightedInfNorm.dispose();
      });

      this.iteration.assign(this.iteration.add(this.one));
      this.accBeta1.assign(this.accBeta1.mul(this.beta1));
    });

    this.variableGradients.dispose();
    this.variableGradients = new TensorArrayMap();
  }

  dispose() {
    super.dispose();
    this.c.dispose();
    this.eps.dispose();
    this.accBeta1.dispose();
    this.beta1.dispose();
    this.beta2.dispose();
    this.oneMinusBeta1.dispose();

    this.decay.dispose();
    this.iteration.dispose();

    this.one.dispose();

    if (this.firstMomentGraph != null) {
      this.firstMomentGraph.dispose();
    }

    if (this.weightedInfNormGraph != null) {
      this.weightedInfNormGraph.dispose();
    }

    if (this.accumulatedFirstMoment != null) {
      Object.keys(this.accumulatedFirstMoment)
          .forEach(name => this.accumulatedFirstMoment[name].dispose());
    }

    if (this.accumulatedWeightedInfNorm != null) {
      Object.keys(this.accumulatedWeightedInfNorm)
          .forEach(name => this.accumulatedWeightedInfNorm[name].dispose());
    }
  }

  // Average of 1st gradient
  private firstMomentGraph = new TensorArrayMap();
  // Average of exponentially weighed infinity norm
  private weightedInfNormGraph = new TensorArrayMap();
}
