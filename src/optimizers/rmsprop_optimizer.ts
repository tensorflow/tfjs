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
import {Scalar} from '../tensor';
import {NamedVariableMap} from '../types';
import {Optimizer} from './optimizer';

/** @doclink Optimizer */
export class RMSPropOptimizer extends Optimizer {
  private c: Scalar;
  private epsilon: Scalar;
  private decay: Scalar;
  private momentum: Scalar;
  private oneMinusDecay: Scalar;
  private centered: boolean;

  private accumulatedMeanSquares: NamedVariableMap = {};
  private accumulatedMeanGrads: NamedVariableMap = {};
  private accumulatedMoments: NamedVariableMap = {};

  constructor(
      protected learningRate: number, decay = 0.9, momentum = 0.0,
      epsilon = 1e-8, centered = false) {
    super();

    this.c = keep(scalar(learningRate));
    this.epsilon = keep(scalar(epsilon));
    this.decay = keep(scalar(decay));
    this.momentum = keep(scalar(momentum));
    this.oneMinusDecay = keep(scalar(1 - decay));
    this.centered = centered;
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
            this.decay.mul(accumulatedMeanSquare)
                .add(this.oneMinusDecay.mul(gradient.square()));

        if (this.centered) {
          // Centered gradient
          const newAccumulatedMeanGrad =
              this.decay.mul(accumulatedMeanGrad)
                  .add(this.oneMinusDecay.mul(gradient));

          const newAccumulatedMoments =
              this.momentum.mul(accumulatedMoments)
                  .add(this.c.mul(gradient).div(
                      newAccumulatedMeanSquare.sub(
                        newAccumulatedMeanGrad.square().add(
                          this.epsilon)).sqrt()));

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
              this.decay.mul(accumulatedMeanSquare)
                  .add(this.oneMinusDecay.mul(gradient.square()));

          const newAccumulatedMoments =
              this.momentum.mul(accumulatedMoments)
                  .add(this.c.mul(gradient).div(
                      newAccumulatedMeanSquare.add(this.epsilon).sqrt()));

          this.accumulatedMeanSquares[variableName].assign(
              newAccumulatedMeanSquare);
          this.accumulatedMoments[variableName].assign(newAccumulatedMoments);

          const newValue = value.sub(newAccumulatedMoments);
          value.assign(newValue);
        }
      });
    }
  }

  dispose() {
    this.c.dispose();
    this.epsilon.dispose();
    this.decay.dispose();
    this.momentum.dispose();
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
}
