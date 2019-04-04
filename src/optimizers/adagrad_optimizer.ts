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
import {fill} from '../ops/ops';
import {ConfigDict, registerClass, Serializable, SerializableConstructor} from '../serialization';
import {NamedVariableMap} from '../tensor_types';
import {Optimizer} from './optimizer';

/** @doclink Optimizer */
export class AdagradOptimizer extends Optimizer {
  /** @nocollapse */
  static className = 'AdagradOptimizer';

  private accumulatedGrads: NamedVariableMap = {};

  constructor(
      protected learningRate: number, private initialAccumulatorValue = 0.1) {
    super();
  }

  applyGradients(variableGradients: NamedVariableMap) {
    for (const variableName in variableGradients) {
      const value = ENGINE.registeredVariables[variableName];
      if (this.accumulatedGrads[variableName] == null) {
        const trainable = false;
        tidy(() => {
          this.accumulatedGrads[variableName] =
              fill(value.shape, this.initialAccumulatorValue)
                  .variable(trainable);
        });
      }

      const gradient = variableGradients[variableName];
      const accumulatedGrad = this.accumulatedGrads[variableName];

      tidy(() => {
        const newAccumulatedGrad = accumulatedGrad.add(gradient.square());
        this.accumulatedGrads[variableName].assign(newAccumulatedGrad);

        const newValue =
            gradient
                .div(newAccumulatedGrad.add(ENGINE.backend.epsilon()).sqrt())
                .mul(-this.learningRate)
                .add(value);
        value.assign(newValue);
      });
    }
  }

  dispose(): void {
    if (this.accumulatedGrads != null) {
      Object.keys(this.accumulatedGrads)
          .forEach(name => this.accumulatedGrads[name].dispose());
    }
  }
  getConfig(): ConfigDict {
    return {
      'learningRate': this.learningRate,
      'initialAccumulatorValue': this.initialAccumulatorValue,
    };
  }

  /** @nocollapse */
  static fromConfig<T extends Serializable>(
      cls: SerializableConstructor<T>, config: ConfigDict): T {
    return new cls(config['learningRate'], config['initialAccumulatorValue']);
  }
}
registerClass(AdagradOptimizer);
