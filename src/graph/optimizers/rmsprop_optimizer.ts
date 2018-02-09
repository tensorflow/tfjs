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

import {keep, tidy} from '../../globals';
import {NDArrayMath} from '../../math/math';
import {scalar} from '../../math/ops';
import {Optimizer} from '../../math/optimizers/optimizer';
import {Scalar, Tensor} from '../../math/tensor';
import {NamedVariableMap} from '../../math/types';
import {Node} from '../graph';
import {SessionRuntime} from '../session';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

export class RMSPropOptimizer extends Optimizer {
  private epsilon: Scalar;
  private gamma: Scalar;

  constructor(
      protected learningRate: number, gamma: number,
      /** @deprecated only for graph */
      specifiedVariableList?: Node[]) {
    super(learningRate, specifiedVariableList);

    this.epsilon = scalar(1e-6);
    this.gamma = scalar(gamma);
    this.one = scalar(1);
  }

  applyGradients(variableGradients: NamedVariableMap) {
    throw new Error(`RMSProp optimizer not implemented for eager.`);
  }

  // Graph
  /** @deprecated only for graph */
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
      });
    }
  }

  /** @deprecated only for graph */
  afterBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    tidy(() => {
      this.variableNodes.forEach(node => {
        const oldVariable = activationArrayMap.get(node.output);
        const gradient = this.variableGradients.get(node.output);
        const oldCache = this.accumulatedSquaredGradientsGraph.get(node.output);

        const gradientSquare = math.multiply(gradient, gradient);
        const cache = math.scaledArrayAdd(
            this.gamma, oldCache, math.subtract(this.one, this.gamma),
            gradientSquare);
        const variable = math.scaledArrayAdd(
            this.cGraph,
            math.divide(gradient, math.add(math.sqrt(cache), this.epsilon)),
            this.one, oldVariable);
        this.accumulatedSquaredGradientsGraph.set(node.output, keep(cache));
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
    this.gamma.dispose();
    if (this.accumulatedSquaredGradientsGraph != null) {
      this.accumulatedSquaredGradientsGraph.dispose();
    }
  }

  private accumulatedSquaredGradientsGraph = new TensorArrayMap();
  private one: Scalar;
}
