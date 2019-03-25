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
export class RMSPropOptimizer extends Optimizer {
  /** @nocollapse */
  static className = 'RMSPropOptimizer';
  private c: Scalar;
  private epsilonScalar: Scalar;
  private decayScalar: Scalar;
  private momentumScalar: Scalar;
  private oneMinusDecay: Scalar;
  private centered: boolean;

  private accumulatedMeanSquares: NamedVariableMap = {};
  private accumulatedMeanGrads: NamedVariableMap = {};
  private accumulatedMoments: NamedVariableMap = {};

  constructor(
      protected learningRate: number, protected decay = 0.9,
      protected momentum = 0.0, protected epsilon: number = null,
      centered = false) {
    super();

    this.c = keep(scalar(learningRate));
    this.decayScalar = keep(scalar(decay));
    this.momentumScalar = keep(scalar(momentum));
    this.oneMinusDecay = keep(scalar(1 - decay));
    this.centered = centered;

    if (epsilon === null) {
      epsilon = ENV.get('EPSILON');
    }

    this.epsilonScalar = keep(scalar(epsilon));
  }

  applyGradients(variableGradients: NamedVariableMap) {
    for (const variableName in variableGradients) {
      const value = ENV.engine.registeredVariables[variableName];
      if (this.accumulatedMeanSquares[variableName] == null) {
        const trainable = false;
        tidy(() => {
          this.accumulatedMeanSquares[variableName] =
              zerosLike(value).variable(trainable);
        });
      }
      if (this.accumulatedMeanGrads[variableName] == null && this.centered) {
        const trainable = false;
        tidy(() => {
          this.accumulatedMeanGrads[variableName] =
              zerosLike(value).variable(trainable);
        });
      }
      if (this.accumulatedMoments[variableName] == null) {
        const trainable = false;
        tidy(() => {
          this.accumulatedMoments[variableName] =
              zerosLike(value).variable(trainable);
        });
      }

      const accumulatedMeanSquare = this.accumulatedMeanSquares[variableName];
      const accumulatedMeanGrad = this.accumulatedMeanGrads[variableName];
      const accumulatedMoments = this.accumulatedMoments[variableName];
      const gradient = variableGradients[variableName];

      tidy(() => {
        const newAccumulatedMeanSquare =
            this.decayScalar.mul(accumulatedMeanSquare)
                .add(this.oneMinusDecay.mul(gradient.square()));

        if (this.centered) {
          // Centered gradient
          const newAccumulatedMeanGrad =
              this.decayScalar.mul(accumulatedMeanGrad)
                  .add(this.oneMinusDecay.mul(gradient));

          const newAccumulatedMoments =
              this.momentumScalar.mul(accumulatedMoments)
                  .add(this.c.mul(gradient).div(
                      newAccumulatedMeanSquare
                          .sub(newAccumulatedMeanGrad.square().add(
                              this.epsilonScalar))
                          .sqrt()));

          this.accumulatedMeanSquares[variableName].assign(
              newAccumulatedMeanSquare);
          this.accumulatedMeanGrads[variableName].assign(
              newAccumulatedMeanGrad);
          this.accumulatedMoments[variableName].assign(newAccumulatedMoments);

          const newValue = value.sub(newAccumulatedMoments);
          value.assign(newValue);
        } else {
          // Plain gradient
          const newAccumulatedMeanSquare =
              this.decayScalar.mul(accumulatedMeanSquare)
                  .add(this.oneMinusDecay.mul(gradient.square()));

          const newAccumulatedMoments =
              this.momentumScalar.mul(accumulatedMoments)
                  .add(this.c.mul(gradient).div(
                      newAccumulatedMeanSquare.add(this.epsilonScalar).sqrt()));

          this.accumulatedMeanSquares[variableName].assign(
              newAccumulatedMeanSquare);
          this.accumulatedMoments[variableName].assign(newAccumulatedMoments);

          const newValue = value.sub(newAccumulatedMoments);
          value.assign(newValue);
        }
      });
    }
  }

  dispose(): void {
    this.c.dispose();
    this.epsilonScalar.dispose();
    this.decayScalar.dispose();
    this.momentumScalar.dispose();
    this.oneMinusDecay.dispose();
    if (this.accumulatedMeanSquares != null) {
      Object.keys(this.accumulatedMeanSquares)
          .forEach(name => this.accumulatedMeanSquares[name].dispose());
    }
    if (this.accumulatedMeanGrads != null && this.centered) {
      Object.keys(this.accumulatedMeanGrads)
          .forEach(name => this.accumulatedMeanGrads[name].dispose());
    }
    if (this.accumulatedMoments != null) {
      Object.keys(this.accumulatedMoments)
          .forEach(name => this.accumulatedMoments[name].dispose());
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
