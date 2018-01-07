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

import {ENV} from '../../environment';
import {Node} from '../../graph/graph';
import {SessionRuntime} from '../../graph/session';
// tslint:disable-next-line:max-line-length
import {SummedTensorArrayMap, TensorArrayMap} from '../../graph/tensor_array_map';
import {NDArrayMath} from '../../math/math';
import {NamedVariableMap} from '../../util';
import {Scalar} from '../ndarray';

import {Optimizer} from './optimizer';

export class SGDOptimizer extends Optimizer {
  private c: Scalar;

  constructor(protected learningRate: number, specifiedVariableList?: Node[]) {
    super(learningRate, specifiedVariableList);
    this.setLearningRate(learningRate);
  }

  // Eager mode
  applyGradients(variableGradients: NamedVariableMap) {
    const math = ENV.math;

    const varNames = Object.keys(variableGradients);
    varNames.forEach(varName => {
      const gradient = variableGradients[varName];
      const value = math.registeredVariables[varName];

      const newValue = math.scope(() => {
        return math.add(math.multiply(this.c, gradient), value);
      });

      value.assign(newValue);
    });
  }

  // Graph
  afterBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    math.scope(keep => {
      this.variableNodes.forEach(node => {
        const oldVariable = activationArrayMap.get(node.output);
        const gradient = this.variableGradients.get(node.output);
        const variable =
            math.scaledArrayAdd(this.cGraph, gradient, this.one, oldVariable);
        activationArrayMap.set(node.output, keep(variable));
        node.data = variable;

        oldVariable.dispose();
      });
    });

    this.variableGradients.dispose();
    this.variableGradients = new TensorArrayMap();
  }

  dispose() {
    this.c.dispose();
    super.dispose();
  }

  setLearningRate(learningRate: number) {
    this.learningRate = learningRate;
    if (this.c != null) {
      this.c.dispose();
    }
    this.c = ENV.math.keep(Scalar.new(-learningRate));
  }
}
