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
import {scalar, zerosLike} from '../ops/ops';
import {Scalar, Tensor} from '../tensor';
import {variable} from '../tensor';
import {NamedVariableMap} from '../types';
import {Optimizer} from './optimizer';

/** @doclink Optimizer */
export class AdadeltaOptimizer extends Optimizer {
  private c: Scalar;
  private epsilon: Scalar;
  private rho: Scalar;
  private oneMinusRho: Scalar;

  private accumulatedGrads: NamedVariableMap = {};
  private accumulatedUpdates: NamedVariableMap = {};

  constructor(
      learningRate: number, rho: number,
      /** @deprecated */
      specifiedVariableList?: Node[], epsilon = 1e-8) {
    super(learningRate, specifiedVariableList);

    this.c = keep(scalar(-learningRate));
    this.epsilon = keep(scalar(epsilon));
    this.rho = keep(scalar(rho));
    this.oneMinusRho = keep(scalar(1 - rho));
  }

  applyGradients(variableGradients: NamedVariableMap) {
    for (const variableName in variableGradients) {
      const value = ENV.engine.registeredVariables[variableName];
      if (this.accumulatedGrads[variableName] == null) {
        const trainable = false;
        tidy(() => {
          this.accumulatedGrads[variableName] =
              variable(zerosLike(value), trainable);
        });
      }
      if (this.accumulatedUpdates[variableName] == null) {
        const trainable = false;
        tidy(() => {
          this.accumulatedUpdates[variableName] =
              variable(zerosLike(value), trainable);
        });
      }

      const gradient = variableGradients[variableName];
      const accumulatedGrad = this.accumulatedGrads[variableName];
      const accumulatedUpdate = this.accumulatedUpdates[variableName];

      tidy(() => {
        const newAccumulatedGrad =
            this.rho.mul(accumulatedGrad)
                .add(this.oneMinusRho.mul(gradient.square()));

        const updates = accumulatedUpdate.add(this.epsilon)
                            .sqrt()
                            .div(accumulatedGrad.add(this.epsilon).sqrt())
                            .mul(gradient);

        const newAccumulatedUpdate =
            this.rho.mul(accumulatedUpdate)
                .add(this.oneMinusRho.mul(updates.square()));

        this.accumulatedGrads[variableName].assign(newAccumulatedGrad);
        this.accumulatedUpdates[variableName].assign(newAccumulatedUpdate);

        const newValue = this.c.mul(updates).add(value);
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
    if (this.accumulatedSquaredGradientsGraph.size() === 0) {
      this.variableNodes.forEach(node => {
        this.accumulatedSquaredGradientsGraph.set(
            node.output, Tensor.zeros(node.output.shape));
        this.accumulatedUpdatesGraph.set(
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
        const oldCache = this.accumulatedSquaredGradientsGraph.get(node.output);
        const oldUpdates = this.accumulatedUpdatesGraph.get(node.output);

        const gradientSquare = math.multiply(gradient, gradient);
        // Exponential decay of average squared gradients.
        const cache = math.scaledArrayAdd(
            this.rho, oldCache, math.subtract(this.one, this.rho),
            gradientSquare);

        const updates = math.multiply(
            math.divide(
                math.sqrt(math.add(oldUpdates, this.epsilon)),
                math.sqrt(math.add(oldCache, this.epsilon))),
            gradient);

        const variable =
            math.scaledArrayAdd(this.cGraph, updates, this.one, oldVariable);

        const updateSquare = math.multiply(updates, updates);
        // Exponential decay of average updated values.
        const newUpdates = math.scaledArrayAdd(
            this.rho, oldUpdates, math.subtract(this.one, this.rho),
            updateSquare);

        this.accumulatedSquaredGradientsGraph.set(node.output, keep(cache));
        this.accumulatedUpdatesGraph.set(node.output, keep(newUpdates));
        activationArrayMap.set(node.output, keep(variable));
        node.data = variable;

        oldVariable.dispose();
        oldCache.dispose();
        oldUpdates.dispose();
      });
    });

    this.variableGradients.dispose();
    this.variableGradients = new TensorArrayMap();
  }

  dispose() {
    super.dispose();
    this.c.dispose();
    this.epsilon.dispose();
    this.rho.dispose();
    this.oneMinusRho.dispose();
    if (this.one != null) {
      this.one.dispose();
    }
    if (this.accumulatedSquaredGradientsGraph != null) {
      this.accumulatedSquaredGradientsGraph.dispose();
    }
    if (this.accumulatedUpdatesGraph != null) {
      this.accumulatedUpdatesGraph.dispose();
    }
    if (this.accumulatedUpdates != null) {
      Object.keys(this.accumulatedUpdates)
          .forEach(name => this.accumulatedUpdates[name].dispose());
      Object.keys(this.accumulatedGrads)
          .forEach(name => this.accumulatedGrads[name].dispose());
    }
  }

  private accumulatedSquaredGradientsGraph = new TensorArrayMap();
  private accumulatedUpdatesGraph = new TensorArrayMap();
  private one: Scalar;
}
