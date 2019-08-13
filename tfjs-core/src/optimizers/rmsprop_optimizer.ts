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
import {zerosLike} from '../ops/ops';
import {ConfigDict, registerClass, Serializable, SerializableConstructor} from '../serialization';
import {NamedTensor, NamedTensorMap} from '../tensor_types';

import {Optimizer, OptimizerVariable} from './optimizer';

/** @doclink Optimizer */
export class RMSPropOptimizer extends Optimizer {
  /** @nocollapse */
  static className = 'RMSProp';  // Note: Name matters for Python compatibility.
  private centered: boolean;

  private accumulatedMeanSquares: OptimizerVariable[] = [];
  private accumulatedMoments: OptimizerVariable[] = [];
  private accumulatedMeanGrads: OptimizerVariable[] = [];

  constructor(
      protected learningRate: number, protected decay = 0.9,
      protected momentum = 0.0, protected epsilon: number = null,
      centered = false) {
    super();

    this.centered = centered;

    if (epsilon == null) {
      this.epsilon = ENGINE.backend.epsilon();
    }
  }

  applyGradients(variableGradients: NamedTensorMap|NamedTensor[]) {
    const variableNames = Array.isArray(variableGradients) ?
        variableGradients.map(item => item.name) :
        Object.keys(variableGradients);

    variableNames.forEach((name, i) => {
      const value = ENGINE.registeredVariables[name];
      const trainable = false;
      if (this.accumulatedMeanSquares[i] == null) {
        this.accumulatedMeanSquares[i] = {
          originalName: `${name}/rms`,
          variable: tidy(() => zerosLike(value).variable(trainable))
        };
      }
      if (this.accumulatedMoments[i] == null) {
        this.accumulatedMoments[i] = {
          originalName: `${name}/momentum`,
          variable: tidy(() => zerosLike(value).variable(trainable))
        };
      }
      if (this.accumulatedMeanGrads[i] == null && this.centered) {
        this.accumulatedMeanGrads[i] = {
          originalName: `${name}/mg`,
          variable: tidy(() => zerosLike(value).variable(trainable))
        };
      }

      const gradient = Array.isArray(variableGradients) ?
          variableGradients[i].tensor :
          variableGradients[name];
      if (gradient == null) {
        return;
      }

      const accumulatedMeanSquare = this.accumulatedMeanSquares[i].variable;
      const accumulatedMoments = this.accumulatedMoments[i].variable;
      tidy(() => {
        const newAccumulatedMeanSquare =
            accumulatedMeanSquare.mul(this.decay)
                .add(gradient.square().mul(1 - this.decay));

        if (this.centered) {
          const accumulatedMeanGrad = this.accumulatedMeanGrads[i].variable;
          // Centered gradient
          const newAccumulatedMeanGrad = accumulatedMeanGrad.mul(this.decay)
                                             .add(gradient.mul(1 - this.decay));

          const newAccumulatedMoments =
              accumulatedMoments.mul(this.momentum)
                  .add(gradient.mul(this.learningRate)
                           .div(newAccumulatedMeanSquare
                                    .sub(newAccumulatedMeanGrad.square().add(
                                        this.epsilon))
                                    .sqrt()));

          accumulatedMeanSquare.assign(newAccumulatedMeanSquare);
          accumulatedMeanGrad.assign(newAccumulatedMeanGrad);
          accumulatedMoments.assign(newAccumulatedMoments);

          const newValue = value.sub(newAccumulatedMoments);
          value.assign(newValue);
        } else {
          // Plain gradient
          const newAccumulatedMeanSquare =
              accumulatedMeanSquare.mul(this.decay)
                  .add(gradient.square().mul(1 - this.decay));

          const newAccumulatedMoments =
              accumulatedMoments.mul(this.momentum)
                  .add(gradient.mul(this.learningRate)
                           .div(newAccumulatedMeanSquare.add(this.epsilon)
                                    .sqrt()));

          accumulatedMeanSquare.assign(newAccumulatedMeanSquare);
          accumulatedMoments.assign(newAccumulatedMoments);

          const newValue = value.sub(newAccumulatedMoments);
          value.assign(newValue);
        }
      });
    });
    this.incrementIterations();
  }

  dispose(): void {
    if (this.accumulatedMeanSquares != null) {
      dispose(this.accumulatedMeanSquares.map(v => v.variable));
    }
    if (this.accumulatedMeanGrads != null && this.centered) {
      dispose(this.accumulatedMeanGrads.map(v => v.variable));
    }
    if (this.accumulatedMoments != null) {
      dispose(this.accumulatedMoments.map(v => v.variable));
    }
  }

  async getWeights(): Promise<NamedTensor[]> {
    // Order matters for Python compatibility.
    const variables: OptimizerVariable[] =
        [...this.accumulatedMeanSquares, ...this.accumulatedMoments];
    if (this.centered) {
      variables.push(...this.accumulatedMeanGrads);
    }
    return [await this.saveIterations()].concat(
        variables.map(v => ({name: v.originalName, tensor: v.variable})));
  }

  async setWeights(weightValues: NamedTensor[]): Promise<void> {
    weightValues = await this.extractIterations(weightValues);
    const variableCount =
        this.centered ? weightValues.length / 3 : weightValues.length / 2;
    const trainable = false;
    this.accumulatedMeanSquares =
        weightValues.slice(0, variableCount).map(v => ({
                                                   originalName: v.name,
                                                   variable: v.tensor.variable(
                                                       trainable)
                                                 }));
    this.accumulatedMoments =
        weightValues.slice(variableCount, variableCount * 2)
            .map(v => ({
                   originalName: v.name,
                   variable: v.tensor.variable(trainable)
                 }));
    if (this.centered) {
      this.accumulatedMeanGrads =
          weightValues.slice(variableCount * 2, variableCount * 3)
              .map(v => ({
                     originalName: v.name,
                     variable: v.tensor.variable(trainable)
                   }));
    }
  }

  getConfig(): ConfigDict {
    return {
      'learningRate': this.learningRate,
      'decay': this.decay,
      'momentum': this.momentum,
      'epsilon': this.epsilon,
      'centered': this.centered
    };
  }

  /** @nocollapse */
  static fromConfig<T extends Serializable>(
      cls: SerializableConstructor<T>, config: ConfigDict): T {
    return new cls(
        config['learningRate'], config['decay'], config['momentum'],
        config['epsilon'], config['centered']);
  }
}
registerClass(RMSPropOptimizer);
