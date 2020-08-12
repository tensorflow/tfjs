/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {ENGINE} from '../engine';
import {dispose, tidy} from '../globals';
import {add} from '../ops/add';
import {div} from '../ops/div';
import {mul} from '../ops/mul';
import {pow} from '../ops/pow';
import {scalar} from '../ops/scalar';
import {sqrt} from '../ops/sqrt';
import {square} from '../ops/square';
import {sub} from '../ops/sub';
import {zerosLike} from '../ops/zeros_like';
import {ConfigDict, registerClass, Serializable, SerializableConstructor} from '../serialization';
import {Variable} from '../tensor';
import {NamedTensor, NamedVariableMap} from '../tensor_types';

import {Optimizer, OptimizerVariable} from './optimizer';

export class AdamOptimizer extends Optimizer {
  /** @nocollapse */
  static className = 'Adam';  // Note: Name matters for Python compatibility.
  private accBeta1: Variable;
  private accBeta2: Variable;

  private accumulatedFirstMoment: OptimizerVariable[] = [];
  private accumulatedSecondMoment: OptimizerVariable[] = [];

  constructor(
      protected learningRate: number, protected beta1: number,
      protected beta2: number, protected epsilon: number = null) {
    super();
    tidy(() => {
      // accB* will be updated by batch.
      this.accBeta1 = scalar(beta1).variable();
      this.accBeta2 = scalar(beta2).variable();
    });

    if (epsilon == null) {
      this.epsilon = ENGINE.backend.epsilon();
    }
  }

  applyGradients(variableGradients: NamedVariableMap|NamedTensor[]) {
    const varNames = Array.isArray(variableGradients) ?
        variableGradients.map(v => v.name) :
        Object.keys(variableGradients);
    tidy(() => {
      const oneMinusAccBeta1 = sub(1, this.accBeta1);
      const oneMinusAccBeta2 = sub(1, this.accBeta2);

      varNames.forEach((name, i) => {
        const value = ENGINE.registeredVariables[name];
        const trainable = false;
        if (this.accumulatedFirstMoment[i] == null) {
          this.accumulatedFirstMoment[i] = {
            originalName: `${name}/m`,
            variable: tidy(() => zerosLike(value).variable(trainable))
          };
        }
        if (this.accumulatedSecondMoment[i] == null) {
          this.accumulatedSecondMoment[i] = {
            originalName: `${name}/v`,
            variable: tidy(() => zerosLike(value).variable(trainable))
          };
        }

        const gradient = Array.isArray(variableGradients) ?
            variableGradients[i].tensor :
            variableGradients[name];
        if (gradient == null) {
          return;
        }

        const firstMoment = this.accumulatedFirstMoment[i].variable;
        const secondMoment = this.accumulatedSecondMoment[i].variable;

        const newFirstMoment =
            add(mul(firstMoment, this.beta1), mul(gradient, 1 - this.beta1));
        const newSecondMoment =
            add(mul(secondMoment, this.beta2),
                mul(square(gradient), 1 - this.beta2));

        const biasCorrectedFirstMoment = div(newFirstMoment, oneMinusAccBeta1);
        const biasCorrectedSecondMoment =
            div(newSecondMoment, oneMinusAccBeta2);

        firstMoment.assign(newFirstMoment);
        secondMoment.assign(newSecondMoment);

        const newValue =
            add(mul(div(biasCorrectedFirstMoment,
                        add(sqrt(biasCorrectedSecondMoment), this.epsilon)),
                    -this.learningRate),
                value);
        value.assign(newValue);
      });

      this.accBeta1.assign(mul(this.accBeta1, this.beta1));
      this.accBeta2.assign(mul(this.accBeta2, this.beta2));
    });
    this.incrementIterations();
  }

  dispose(): void {
    this.accBeta1.dispose();
    this.accBeta2.dispose();

    if (this.accumulatedFirstMoment != null) {
      dispose(this.accumulatedFirstMoment.map(v => v.variable));
    }
    if (this.accumulatedSecondMoment != null) {
      dispose(this.accumulatedSecondMoment.map(v => v.variable));
    }
  }

  async getWeights(): Promise<NamedTensor[]> {
    // Order matters for Python compatibility.
    const variables: OptimizerVariable[] =
        [...this.accumulatedFirstMoment, ...this.accumulatedSecondMoment];
    return [await this.saveIterations()].concat(
        variables.map(v => ({name: v.originalName, tensor: v.variable})));
  }

  async setWeights(weightValues: NamedTensor[]): Promise<void> {
    weightValues = await this.extractIterations(weightValues);
    tidy(() => {
      this.accBeta1.assign(pow(this.beta1, this.iterations_ + 1));
      this.accBeta2.assign(pow(this.beta2, this.iterations_ + 1));
    });

    const variableCount = weightValues.length / 2;
    const trainable = false;
    this.accumulatedFirstMoment =
        weightValues.slice(0, variableCount).map(v => ({
                                                   originalName: v.name,
                                                   variable: v.tensor.variable(
                                                       trainable)
                                                 }));
    this.accumulatedSecondMoment =
        weightValues.slice(variableCount, variableCount * 2)
            .map(v => ({
                   originalName: v.name,
                   variable: v.tensor.variable(trainable)
                 }));
  }

  getConfig(): ConfigDict {
    return {
      'learningRate': this.learningRate,
      'beta1': this.beta1,
      'beta2': this.beta2,
      'epsilon': this.epsilon,
    };
  }

  /** @nocollapse */
  static fromConfig<T extends Serializable>(
      cls: SerializableConstructor<T>, config: ConfigDict): T {
    return new cls(
        config['learningRate'], config['beta1'], config['beta2'],
        config['epsilon']);
  }
}
registerClass(AdamOptimizer);
