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
import {NamedVariableMap} from '../types';

import {SGDOptimizer} from './sgd_optimizer';

/** @doclink Optimizer */
export class MomentumOptimizer extends SGDOptimizer {
  private m: Scalar;
  private accumulations: NamedVariableMap;

  constructor(
      protected learningRate: number, private momentum: number,
      specifiedVariableList?: Node[], private useNesterov = false) {
    super(learningRate, specifiedVariableList);
    this.m = scalar(this.momentum);
    this.accumulations = {};
  }

  applyGradients(variableGradients: NamedVariableMap) {
    for (const variableName in variableGradients) {
      const value = ENV.engine.registeredVariables[variableName];
      if (this.accumulations[variableName] == null) {
        const trainable = false;
        tidy(() => {
          this.accumulations[variableName] =
              zerosLike(value).variable(trainable);
        });
      }

      const accumulation = this.accumulations[variableName];
      const gradient = variableGradients[variableName];

      tidy(() => {
        let newValue: Tensor;
        const newAccumulation = this.m.mul(accumulation).add(gradient);
        if (this.useNesterov) {
          newValue =
              this.c.mul(gradient.add(newAccumulation.mul(this.m))).add(value);
        } else {
          newValue = this.c.mul(newAccumulation).add(value);
        }
        this.accumulations[variableName].assign(newAccumulation);
        value.assign(newValue);
      });
    }
  }

  beforeBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    if (this.variableVelocitiesGraph == null) {
      this.variableVelocitiesGraph = new TensorArrayMap();
    }

    super.beforeBatch(
        math, batchSize, runtime, activationArrayMap, gradientArrayMap);

    if (this.variableVelocitiesGraph.size() === 0) {
      this.variableNodes.forEach(node => {
        this.variableVelocitiesGraph.set(
            node.output, Tensor.zeros(node.output.shape));
      });
    }
  }

  afterBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    tidy(() => {
      this.variableNodes.forEach(node => {
        const oldVariable = activationArrayMap.get(node.output);
        const gradient = this.variableGradients.get(node.output);
        const oldVelocity = this.variableVelocitiesGraph.get(node.output);

        let variable: Tensor;
        const velocity = this.m.mul(oldVelocity).add(gradient);
        if (this.useNesterov) {
          variable = this.cGraph.mul(gradient.add(velocity.mul(this.m)))
                         .add(oldVariable);
        } else {
          variable = this.cGraph.mul(velocity).add(oldVariable);
        }

        this.variableVelocitiesGraph.set(node.output, keep(velocity));
        activationArrayMap.set(node.output, keep(variable));
        node.data = variable;

        oldVariable.dispose();
        oldVelocity.dispose();
      });
    });

    this.variableGradients.dispose();
    this.variableGradients = new TensorArrayMap();
  }

  dispose() {
    super.dispose();
    this.m.dispose();
    if (this.variableVelocitiesGraph != null) {
      this.variableVelocitiesGraph.dispose();
    }
    if (this.accumulations != null) {
      for (const variableName in this.accumulations) {
        this.accumulations[variableName].dispose();
      }
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

  // Graph.
  private variableVelocitiesGraph: TensorArrayMap;
}
