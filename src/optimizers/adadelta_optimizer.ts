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
import {zerosLike} from '../ops/ops';
import {ConfigDict, registerClass, Serializable, SerializableConstructor} from '../serialization';
import {NamedVariableMap} from '../tensor_types';
import {Optimizer} from './optimizer';

/** @doclink Optimizer */
export class AdadeltaOptimizer extends Optimizer {
  /** @nocollapse */
  static className = 'AdadeltaOptimizer';
  private accumulatedGrads: NamedVariableMap = {};
  private accumulatedUpdates: NamedVariableMap = {};

  constructor(
      protected learningRate: number, protected rho: number,
      protected epsilon: number = null) {
    super();

    if (epsilon == null) {
      this.epsilon = ENGINE.backend.epsilon();
    }
  }

  applyGradients(variableGradients: NamedVariableMap) {
    for (const variableName in variableGradients) {
      const value = ENGINE.registeredVariables[variableName];
      if (this.accumulatedGrads[variableName] == null) {
        const trainable = false;
        tidy(() => {
          this.accumulatedGrads[variableName] =
              zerosLike(value).variable(trainable);
        });
      }
      if (this.accumulatedUpdates[variableName] == null) {
        const trainable = false;
        tidy(() => {
          this.accumulatedUpdates[variableName] =
              zerosLike(value).variable(trainable);
        });
      }

      const gradient = variableGradients[variableName];
      const accumulatedGrad = this.accumulatedGrads[variableName];
      const accumulatedUpdate = this.accumulatedUpdates[variableName];

      tidy(() => {
        const newAccumulatedGrad = accumulatedGrad.mul(this.rho).add(
            gradient.square().mul(1 - this.rho));

        const updates = accumulatedUpdate.add(this.epsilon)
                            .sqrt()
                            .div(accumulatedGrad.add(this.epsilon).sqrt())
                            .mul(gradient);

        const newAccumulatedUpdate = accumulatedUpdate.mul(this.rho).add(
            updates.square().mul(1 - this.rho));

        this.accumulatedGrads[variableName].assign(newAccumulatedGrad);
        this.accumulatedUpdates[variableName].assign(newAccumulatedUpdate);

        const newValue = updates.mul(-this.learningRate).add(value);
        value.assign(newValue);
      });
    }
  }

  dispose(): void {
    if (this.accumulatedUpdates != null) {
      Object.keys(this.accumulatedUpdates)
          .forEach(name => this.accumulatedUpdates[name].dispose());
      Object.keys(this.accumulatedGrads)
          .forEach(name => this.accumulatedGrads[name].dispose());
    }
  }
  getConfig(): ConfigDict {
    return {
      'learningRate': this.learningRate,
      'rho': this.rho,
      'epsilon': this.epsilon
    };
  }

  /** @nocollapse */
  static fromConfig<T extends Serializable>(
      cls: SerializableConstructor<T>, config: ConfigDict): T {
    return new cls(config['learningRate'], config['rho'], config['epsilon']);
  }
}
registerClass(AdadeltaOptimizer);
