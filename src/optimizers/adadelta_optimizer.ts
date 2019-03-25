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
import {scalar, zerosLike} from '../ops/ops';
import {ConfigDict, registerClass, Serializable, SerializableConstructor} from '../serialization';
import {Scalar} from '../tensor';
import {NamedVariableMap} from '../tensor_types';
import {Optimizer} from './optimizer';

/** @doclink Optimizer */
export class AdadeltaOptimizer extends Optimizer {
  /** @nocollapse */
  static className = 'AdadeltaOptimizer';
  private c: Scalar;
  private epsilonScalar: Scalar;
  private rhoScalar: Scalar;
  private oneMinusRho: Scalar;

  private accumulatedGrads: NamedVariableMap = {};
  private accumulatedUpdates: NamedVariableMap = {};

  constructor(
      protected learningRate: number, protected rho: number,
      protected epsilon: number = null) {
    super();

    this.c = keep(scalar(-learningRate));
    this.rhoScalar = keep(scalar(rho));
    this.oneMinusRho = keep(scalar(1 - rho));

    if (epsilon === null) {
      epsilon = ENV.get('EPSILON');
    }

    this.epsilonScalar = keep(scalar(epsilon));
  }

  applyGradients(variableGradients: NamedVariableMap) {
    for (const variableName in variableGradients) {
      const value = ENV.engine.registeredVariables[variableName];
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
        const newAccumulatedGrad =
            this.rhoScalar.mul(accumulatedGrad)
                .add(this.oneMinusRho.mul(gradient.square()));

        const updates = accumulatedUpdate.add(this.epsilonScalar)
                            .sqrt()
                            .div(accumulatedGrad.add(this.epsilonScalar).sqrt())
                            .mul(gradient);

        const newAccumulatedUpdate =
            this.rhoScalar.mul(accumulatedUpdate)
                .add(this.oneMinusRho.mul(updates.square()));

        this.accumulatedGrads[variableName].assign(newAccumulatedGrad);
        this.accumulatedUpdates[variableName].assign(newAccumulatedUpdate);

        const newValue = this.c.mul(updates).add(value);
        value.assign(newValue);
      });
    }
  }

  dispose(): void {
    this.c.dispose();
    this.epsilonScalar.dispose();
    this.rhoScalar.dispose();
    this.oneMinusRho.dispose();
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
