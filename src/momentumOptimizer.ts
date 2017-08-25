/* Copyright 2017 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {Node} from './graph';
import {NDArrayMath} from './math/math';
import {NDArray, Scalar} from './math/ndarray';
import {SGDOptimizer} from './sgd_optimizer';
import {SessionRuntime} from './session';
import {TensorArrayMap} from './tensor_array_map';

export class MomentumOptimizer extends SGDOptimizer {
  constructor(protected learningRate: number, 
    private momentum: number, specifiedVariableList?: Node[]) {
    super(learningRate, specifiedVariableList);
  }

  beforeBatch(
    math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
    activationArrayMap: TensorArrayMap, gradientArrayMap: TensorArrayMap) {
    super.beforeBatch(math, batchSize, runtime,
      activationArrayMap, gradientArrayMap);

    this.m = Scalar.new(this.momentum);
    if (this.variableVelocities.size() === 0) {
      this.variableNodes.forEach(node => {
        this.variableVelocities.set(node.output,
          NDArray.zeros(node.output.shape));
      });
    }
  }

  afterBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap, gradientArrayMap: TensorArrayMap) {
    math.scope((keep) => {
      this.variableNodes.forEach(node => {
        const oldVariable = activationArrayMap.get(node.output);
        const gradient = this.variableGradients.get(node.output);
        const oldVelocity = this.variableVelocities.get(node.output);
        const velocity = math.scaledArrayAdd(this.m,
            oldVelocity, this.one, gradient);
        const variable =
            math.scaledArrayAdd(this.c!, velocity, this.one!, oldVariable);
        this.variableVelocities.set(node.output, keep(velocity));
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
    if (this.c != null) {
      this.c.dispose();
    }
    if (this.m != null) {
      this.m.dispose();
    }
    this.one.dispose();
    this.variableVelocities.dispose();
  }

  setMomentum(momentum: number) {
    this.momentum = momentum;
  }

  private variableVelocities = new TensorArrayMap();
  private m: Scalar;
}