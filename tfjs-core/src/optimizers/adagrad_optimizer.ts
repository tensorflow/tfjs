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
import {dispose, tidy} from '../globals';
import {fill} from '../ops/ops';
import {ConfigDict, registerClass, Serializable, SerializableConstructor} from '../serialization';
import {NamedTensor, NamedVariableMap} from '../tensor_types';

import {Optimizer, OptimizerVariable} from './optimizer';

/** @doclink Optimizer */
export class AdagradOptimizer extends Optimizer {
  /** @nocollapse */
  static className = 'Adagrad';  // Note: Name matters for Python compatibility.

  private accumulatedGrads: OptimizerVariable[] = [];

  constructor(
      protected learningRate: number, private initialAccumulatorValue = 0.1) {
    super();
  }

  applyGradients(variableGradients: NamedVariableMap|NamedTensor[]) {
    const variableNames = Array.isArray(variableGradients) ?
        variableGradients.map(item => item.name) :
        Object.keys(variableGradients);

    variableNames.forEach((name, i) => {
      const value = ENGINE.registeredVariables[name];
      if (this.accumulatedGrads[i] == null) {
        const trainable = false;
        this.accumulatedGrads[i] = {
          originalName: `${name}/accumulator`,
          variable: tidy(
              () => fill(value.shape, this.initialAccumulatorValue)
                        .variable(trainable))
        };
      }

      const gradient = Array.isArray(variableGradients) ?
          variableGradients[i].tensor :
          variableGradients[name];
      if (gradient == null) {
        return;
      }

      const accumulatedGrad = this.accumulatedGrads[i].variable;

      tidy(() => {
        const newAccumulatedGrad = accumulatedGrad.add(gradient.square());
        accumulatedGrad.assign(newAccumulatedGrad);

        const newValue =
            gradient
                .div(newAccumulatedGrad.add(ENGINE.backend.epsilon()).sqrt())
                .mul(-this.learningRate)
                .add(value);
        value.assign(newValue);
      });
    });
    this.incrementIterations();
  }

  dispose(): void {
    if (this.accumulatedGrads != null) {
      dispose(this.accumulatedGrads.map(v => v.variable));
    }
  }

  async getWeights(): Promise<NamedTensor[]> {
    // Order matters for Python compatibility.
    return [await this.saveIterations()].concat(this.accumulatedGrads.map(
        v => ({name: v.originalName, tensor: v.variable})));
  }

  async setWeights(weightValues: NamedTensor[]): Promise<void> {
    weightValues = await this.extractIterations(weightValues);
    const trainable = false;
    this.accumulatedGrads = weightValues.map(
        v => ({originalName: v.name, variable: v.tensor.variable(trainable)}));
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
