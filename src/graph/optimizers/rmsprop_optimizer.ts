/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {NDArrayMath} from '../../math/math';
import {NDArray, Scalar} from '../../math/ndarray';
import {Node} from '../graph';
import {SessionRuntime} from '../session';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

import {Optimizer} from './optimizer';

export class RMSPropOptimizer extends Optimizer {
  constructor(
      protected learningRate: number,
      private gamma: number, specifiedVariableList?: Node[]) {
    super(learningRate, specifiedVariableList);
    this.eps = Scalar.new(1e-6);
    this.g = Scalar.new(this.gamma);
  }

  beforeBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    super.beforeBatch(
        math, batchSize, runtime, activationArrayMap, gradientArrayMap);
    if (this.accumulatedSquaredGradients.size() === 0) {
      this.variableNodes.forEach(node => {
        this.accumulatedSquaredGradients.set(
            node.output, NDArray.zeros(node.output.shape));
      });
    }
  }

  afterBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    math.scope((keep) => {
      this.variableNodes.forEach(node => {
        const oldVariable = activationArrayMap.get(node.output);
        const gradient = this.variableGradients.get(node.output);
        const oldCache = this.accumulatedSquaredGradients.get(node.output);
        const gradientSquare = math.multiply(gradient, gradient);
        const cache = math.scaledArrayAdd(
            this.g, oldCache, math.sub(this.one, this.g), gradientSquare);
        const variable = math.scaledArrayAdd(
            this.c, math.divide(gradient, math.add(math.sqrt(cache), this.eps)),
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
    this.eps.dispose();
    this.g.dispose();
    this.accumulatedSquaredGradients.dispose();
  }

  private accumulatedSquaredGradients = new TensorArrayMap();
  private eps: Scalar;
  private g: Scalar;
}
