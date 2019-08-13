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
import {keep, tidy} from '../globals';
import {scalar} from '../ops/ops';
import {ConfigDict, registerClass, Serializable, SerializableConstructor} from '../serialization';
import {Scalar} from '../tensor';
import {NamedTensor, NamedTensorMap} from '../tensor_types';

import {Optimizer} from './optimizer';

/** @doclink Optimizer */
export class SGDOptimizer extends Optimizer {
  /** @nocollapse */
  static className = 'SGD';  // Note: Name matters for Python compatibility.
  protected c: Scalar;

  constructor(protected learningRate: number) {
    super();
    this.setLearningRate(learningRate);
  }

  applyGradients(variableGradients: NamedTensorMap|NamedTensor[]) {
    const varNames = Array.isArray(variableGradients) ?
        variableGradients.map(v => v.name) :
        Object.keys(variableGradients);
    varNames.forEach((name, i) => {
      const gradient = Array.isArray(variableGradients) ?
          variableGradients[i].tensor :
          variableGradients[name];
      if (gradient == null) {
        return;
      }
      const value = ENGINE.registeredVariables[name];
      tidy(() => {
        const newValue = this.c.mul(gradient).add(value);
        value.assign(newValue);
      });
    });
    this.incrementIterations();
  }

  /**
   * Sets the learning rate of the optimizer.
   */
  setLearningRate(learningRate: number) {
    this.learningRate = learningRate;
    if (this.c != null) {
      this.c.dispose();
    }
    this.c = keep(scalar(-learningRate));
  }

  dispose() {
    this.c.dispose();
  }

  async getWeights(): Promise<NamedTensor[]> {
    return [await this.saveIterations()];
  }

  async setWeights(weightValues: NamedTensor[]): Promise<void> {
    weightValues = await this.extractIterations(weightValues);
    if (weightValues.length !== 0) {
      throw new Error('SGD optimizer does not have settable weights.');
    }
  }

  getConfig(): ConfigDict {
    return {'learningRate': this.learningRate};
  }

  /** @nocollapse */
  static fromConfig<T extends Serializable>(
      cls: SerializableConstructor<T>, config: ConfigDict): T {
    return new cls(config['learningRate']);
  }
}
registerClass(SGDOptimizer);
