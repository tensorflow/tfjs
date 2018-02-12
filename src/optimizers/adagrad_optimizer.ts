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
import {Node} from '../graph/graph';
import {SessionRuntime} from '../graph/session';
// tslint:disable-next-line:max-line-length
import {SummedTensorArrayMap, TensorArrayMap} from '../graph/tensor_array_map';
import {NDArrayMath} from '../math';
import {Optimizer} from './optimizer';
import {Scalar, Tensor} from '../tensor';
import {NamedVariableMap} from '../types';
import {fill, scalar} from '../ops/ops';
import {variable} from '../tensor';

export class AdagradOptimizer extends Optimizer {
  private c: Scalar;
  private epsilon: Scalar;

  private accumulatedGrads: NamedVariableMap = {};

  constructor(
      protected learningRate: number, specifiedVariableList?: Node[],
      private initialAccumulatorValue = 0.1) {
    super(learningRate, specifiedVariableList);

    this.c = keep(scalar(-learningRate));
    this.epsilon = keep(scalar(1e-8));
  }

  applyGradients(variableGradients: NamedVariableMap) {
    for (const variableName in variableGradients) {
      const value = ENV.engine.registeredVariables[variableName];
      if (this.accumulatedGrads[variableName] == null) {
        const trainable = false;
        this.accumulatedGrads[variableName] = variable(
            fill(value.shape, this.initialAccumulatorValue), trainable);
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

  /** @deprecated */
  beforeBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    super.beforeBatch(
        math, batchSize, runtime, activationArrayMap, gradientArrayMap);

    if (this.accumulatedSquaredGradients.size() === 0) {
      this.variableNodes.forEach(node => {
        this.accumulatedSquaredGradients.set(
            node.output, Tensor.zeros(node.output.shape));
      });
    }
  }

  /** @deprecated */
  afterBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    if (this.one == null) {
      this.one = keep(scalar(1));
    }
    tidy(() => {
      this.variableNodes.forEach(node => {
        const oldVariable = activationArrayMap.get(node.output);
        const gradient = this.variableGradients.get(node.output);
        const oldCache = this.accumulatedSquaredGradients.get(node.output);

        const gradientSquare = math.multiply(gradient, gradient);
        const cache = math.add(oldCache, gradientSquare);
        const variable = math.scaledArrayAdd(
            this.cGraph,
            math.divide(gradient, math.add(math.sqrt(cache), this.epsilon)),
            this.one, oldVariable);
        this.accumulatedSquaredGradients.set(node.output, keep(cache));
        activationArrayMap.set(node.output, keep(variable));
        node.data = variable;
        oldVariable.dispose();
        oldCache.dispose();
      });
    });

    this.variableGradients.dispose();
    this.variableGradients = new TensorArrayMap();
  }

  dispose() {
    super.dispose();
    this.epsilon.dispose();
    this.c.dispose();
    if (this.one != null) {
      this.one.dispose();
    }
    if (this.accumulatedSquaredGradients != null) {
      this.accumulatedSquaredGradients.dispose();
    }
    if (this.accumulatedGrads != null) {
      Object.keys(this.accumulatedGrads)
          .forEach(name => this.accumulatedGrads[name].dispose());
    }
  }

  private accumulatedSquaredGradients = new TensorArrayMap();
  private one: Scalar;
}
