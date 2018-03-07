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
import {fill, scalar} from '../ops/ops';
import {Scalar} from '../tensor';
import {NamedVariableMap} from '../types';

import {Optimizer} from './optimizer';

/** @doclink Optimizer */
export class AdagradOptimizer extends Optimizer {
  private c: Scalar;
  private epsilon: Scalar;

  private accumulatedGrads: NamedVariableMap = {};

  constructor(
      protected learningRate: number, private initialAccumulatorValue = 0.1) {
    super();
    this.c = keep(scalar(-learningRate));
    this.epsilon = keep(scalar(1e-8));
  }

  applyGradients(variableGradients: NamedVariableMap) {
    for (const variableName in variableGradients) {
      const value = ENV.engine.registeredVariables[variableName];
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
            this.c
                .mul(gradient.div(newAccumulatedGrad.add(this.epsilon).sqrt()))
                .add(value);
        value.assign(newValue);
      });
    }
  }

  dispose() {
    this.epsilon.dispose();
    this.c.dispose();
    if (this.accumulatedGrads != null) {
      Object.keys(this.accumulatedGrads)
          .forEach(name => this.accumulatedGrads[name].dispose());
    }
  }
}
