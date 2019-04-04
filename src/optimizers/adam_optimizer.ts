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

import {ENGINE} from '../engine';
import {tidy} from '../globals';
import {scalar, sub, zerosLike} from '../ops/ops';
import {ConfigDict, registerClass, Serializable, SerializableConstructor} from '../serialization';
import {Variable} from '../tensor';
import {NamedVariableMap} from '../tensor_types';
import {Optimizer} from './optimizer';

export class AdamOptimizer extends Optimizer {
  /** @nocollapse */
  static className = 'AdamOptimizer';
  private accBeta1: Variable;
  private accBeta2: Variable;

  private accumulatedFirstMoment: NamedVariableMap = {};
  private accumulatedSecondMoment: NamedVariableMap = {};

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

  applyGradients(variableGradients: NamedVariableMap) {
    tidy(() => {
      const oneMinusAccBeta1 = sub(1, this.accBeta1);
      const oneMinusAccBeta2 = sub(1, this.accBeta2);

      for (const variableName in variableGradients) {
        const value = ENGINE.registeredVariables[variableName];
        if (this.accumulatedFirstMoment[variableName] == null) {
          const trainable = false;
          this.accumulatedFirstMoment[variableName] =
              zerosLike(value).variable(trainable);
        }
        if (this.accumulatedSecondMoment[variableName] == null) {
          const trainable = false;
          this.accumulatedSecondMoment[variableName] =
              zerosLike(value).variable(trainable);
        }

        const gradient = variableGradients[variableName];
        const firstMoment = this.accumulatedFirstMoment[variableName];
        const secondMoment = this.accumulatedSecondMoment[variableName];

        const newFirstMoment =
            firstMoment.mul(this.beta1).add(gradient.mul(1 - this.beta1));
        const newSecondMoment = secondMoment.mul(this.beta2)
                                    .add(gradient.square().mul(1 - this.beta2));

        const biasCorrectedFirstMoment = newFirstMoment.div(oneMinusAccBeta1);
        const biasCorrectedSecondMoment = newSecondMoment.div(oneMinusAccBeta2);

        this.accumulatedFirstMoment[variableName].assign(newFirstMoment);
        this.accumulatedSecondMoment[variableName].assign(newSecondMoment);

        const newValue =
            biasCorrectedFirstMoment
                .div(biasCorrectedSecondMoment.sqrt().add(this.epsilon))
                .mul(-this.learningRate)
                .add(value);
        value.assign(newValue);
      }

      this.accBeta1.assign(this.accBeta1.mul(this.beta1));
      this.accBeta2.assign(this.accBeta2.mul(this.beta2));
    });
  }

  dispose(): void {
    this.accBeta1.dispose();
    this.accBeta2.dispose();

    if (this.accumulatedFirstMoment != null) {
      Object.keys(this.accumulatedFirstMoment)
          .forEach(name => this.accumulatedFirstMoment[name].dispose());
    }

    if (this.accumulatedSecondMoment != null) {
      Object.keys(this.accumulatedSecondMoment)
          .forEach(name => this.accumulatedSecondMoment[name].dispose());
    }
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
