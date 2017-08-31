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
import {Optimizer} from './optimizer';
import {SessionRuntime} from './session';
import * as session_util from './session_util';
import {TensorArrayMap, SummedTensorArrayMap} from './tensor_array_map';

export class SGDOptimizer extends Optimizer {
  constructor(protected learningRate: number, specifiedVariableList?: Node[]) {
    super(specifiedVariableList);
  }

  beforeBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    this.variableNodes = this.specifiedVariableNodes == null ?
        session_util.getVariableNodesFromEvaluationSet(runtime.nodes) :
        this.specifiedVariableNodes;
    if (batchSize !== this.prevBatchSize) {
      this.prevBatchSize = batchSize;
      this.c = Scalar.new(-this.learningRate / batchSize);
    }
    this.variableNodes.forEach(
        node => this.variableGradients.set(
            node.output, NDArray.zeros(node.output.shape)));
  }

  afterExample(
      math: NDArrayMath, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    math.scope((keep) => {
      this.variableNodes.forEach(node => {
        const gradient = gradientArrayMap.get(node.output);
        const accumulatedGradient = this.variableGradients.get(node.output);
        this.variableGradients.set(
            node.output, keep(math.add(gradient, accumulatedGradient)));
        accumulatedGradient.dispose();
      });
    });
  }

  afterBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    math.scope((keep) => {
      this.variableNodes.forEach(node => {
        const oldVariable = activationArrayMap.get(node.output);
        const gradient = this.variableGradients.get(node.output);
        const variable =
            math.scaledArrayAdd(this.c, gradient, this.one, oldVariable);
        activationArrayMap.set(node.output, keep(variable));
        node.data = variable;

        oldVariable.dispose();
      });
    });

    this.variableGradients.dispose();
    this.variableGradients = new TensorArrayMap();
  }

  dispose() {
    if (this.c != null) {
      this.c.dispose();
    }
    this.one.dispose();
  }

  setLearningRate(learningRate: number) {
    this.learningRate = learningRate;
  }

  protected variableGradients = new TensorArrayMap();
  protected prevBatchSize: number;
  protected one = Scalar.new(1);
  protected c: Scalar;
}
