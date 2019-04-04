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
import {div, scalar, sub, zerosLike} from '../ops/ops';
import {ConfigDict, registerClass, Serializable, SerializableConstructor} from '../serialization';
import {Variable} from '../tensor';
import {NamedVariableMap} from '../tensor_types';
import {Optimizer} from './optimizer';

export class AdamaxOptimizer extends Optimizer {
  /** @nocollapse */
  static className = 'AdamaxOptimizer';
  private accBeta1: Variable;
  private iteration: Variable;

  private accumulatedFirstMoment: NamedVariableMap = {};
  private accumulatedWeightedInfNorm: NamedVariableMap = {};

  constructor(
      protected learningRate: number, protected beta1: number,
      protected beta2: number, protected epsilon: number = null,
      protected decay = 0.0) {
    super();

    tidy(() => {
      this.iteration = scalar(0).variable();
      this.accBeta1 = scalar(beta1).variable();
    });

    if (epsilon == null) {
      this.epsilon = ENGINE.backend.epsilon();
    }
  }

  applyGradients(variableGradients: NamedVariableMap) {
    tidy(() => {
      const oneMinusAccBeta1 = sub(1, this.accBeta1);
      const lr = div(-this.learningRate, this.iteration.mul(this.decay).add(1));

      for (const variableName in variableGradients) {
        const value = ENGINE.registeredVariables[variableName];
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
            firstMoment.mul(this.beta1).add(gradient.mul(1 - this.beta1));

        const ut0 = weightedInfNorm.mul(this.beta2);
        const ut1 = gradient.abs();

        const newWeightedInfNorm = ut0.maximum(ut1);

        this.accumulatedFirstMoment[variableName].assign(newFirstMoment);
        this.accumulatedWeightedInfNorm[variableName].assign(
            newWeightedInfNorm);

        const newValue =
            lr.div(oneMinusAccBeta1)
                .mul(newFirstMoment.div(newWeightedInfNorm.add(this.epsilon)))
                .add(value);

        value.assign(newValue);
      }

      this.iteration.assign(this.iteration.add(1));
      this.accBeta1.assign(this.accBeta1.mul(this.beta1));
    });
  }

  dispose(): void {
    this.accBeta1.dispose();
    this.iteration.dispose();

    if (this.accumulatedFirstMoment != null) {
      Object.keys(this.accumulatedFirstMoment)
          .forEach(name => this.accumulatedFirstMoment[name].dispose());
    }

    if (this.accumulatedWeightedInfNorm != null) {
      Object.keys(this.accumulatedWeightedInfNorm)
          .forEach(name => this.accumulatedWeightedInfNorm[name].dispose());
    }
  }
  getConfig(): ConfigDict {
    return {
      'learningRate': this.learningRate,
      'beta1': this.beta1,
      'beta2': this.beta2,
      'epsilon': this.epsilon,
      'decay': this.decay
    };
  }

  /** @nocollapse */
  static fromConfig<T extends Serializable>(
      cls: SerializableConstructor<T>, config: ConfigDict): T {
    return new cls(
        config['learningRate'], config['beta1'], config['beta2'],
        config['epsilon'], config['decay']);
  }
}
registerClass(AdamaxOptimizer);
