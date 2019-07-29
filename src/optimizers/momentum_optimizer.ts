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
import {scalar, zerosLike} from '../ops/ops';
import {ConfigDict, registerClass, Serializable, SerializableConstructor} from '../serialization';
import {Scalar, Tensor} from '../tensor';
import {NamedTensor, NamedVariableMap} from '../tensor_types';

import {OptimizerVariable} from './optimizer';
import {SGDOptimizer} from './sgd_optimizer';

/** @doclink Optimizer */
export class MomentumOptimizer extends SGDOptimizer {
  /** @nocollapse */
  static className = 'Momentum';  // Name matters for Python compatibility.
  private m: Scalar;
  private accumulations: OptimizerVariable[] = [];

  constructor(
      protected learningRate: number, private momentum: number,
      private useNesterov = false) {
    super(learningRate);
    this.m = scalar(this.momentum);
  }

  applyGradients(variableGradients: NamedVariableMap|NamedTensor[]) {
    const variableNames = Array.isArray(variableGradients) ?
        variableGradients.map(item => item.name) :
        Object.keys(variableGradients);

    variableNames.forEach((name, i) => {
      const value = ENGINE.registeredVariables[name];
      if (this.accumulations[i] == null) {
        const trainable = false;
        this.accumulations[i] = {
          originalName: `${name}/momentum`,
          variable: tidy(() => zerosLike(value).variable(trainable))
        };
      }

      const accumulation = this.accumulations[i].variable;
      const gradient = Array.isArray(variableGradients) ?
          variableGradients[i].tensor :
          variableGradients[name];
      if (gradient == null) {
        return;
      }

      tidy(() => {
        let newValue: Tensor;
        const newAccumulation = this.m.mul(accumulation).add(gradient);
        if (this.useNesterov) {
          newValue =
              this.c.mul(gradient.add(newAccumulation.mul(this.m))).add(value);
        } else {
          newValue = this.c.mul(newAccumulation).add(value);
        }
        accumulation.assign(newAccumulation);
        value.assign(newValue);
      });
    });
    this.incrementIterations();
  }

  dispose(): void {
    this.m.dispose();
    if (this.accumulations != null) {
      dispose(this.accumulations.map(v => v.variable));
    }
  }

  /**
   * Sets the momentum of the optimizer.
   *
   * @param momentum
   */
  setMomentum(momentum: number) {
    this.momentum = momentum;
  }

  async getWeights(): Promise<NamedTensor[]> {
    // Order matters for Python compatibility.
    return [await this.saveIterations()].concat(this.accumulations.map(
        v => ({name: v.originalName, tensor: v.variable})));
  }

  async setWeights(weightValues: NamedTensor[]): Promise<void> {
    weightValues = await this.extractIterations(weightValues);
    const trainable = false;
    this.accumulations = weightValues.map(
        v => ({originalName: v.name, variable: v.tensor.variable(trainable)}));
  }

  getConfig(): ConfigDict {
    return {
      'learningRate': this.learningRate,
      'momentum': this.momentum,
      'useNesterov': this.useNesterov
    };
  }

  /** @nocollapse */
  static fromConfig<T extends Serializable>(
      cls: SerializableConstructor<T>, config: ConfigDict): T {
    return new cls(
        config['learningRate'], config['momentum'], config['useNesterov']);
  }
}
registerClass(MomentumOptimizer);
