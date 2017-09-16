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

import {Node, VariableNode} from './graph';
import {NDArrayMath} from './math/math';
import {NDArray, Scalar} from './math/ndarray';
import {SessionRuntime} from './session';
import * as session_util from './session_util';
import {SummedTensorArrayMap, TensorArrayMap} from './tensor_array_map';

export abstract class Optimizer {
  protected variableNodes: VariableNode[];
  protected specifiedVariableNodes: VariableNode[]|null;

  constructor(protected learningRate: number, specifiedVariableList?: Node[]) {
    if (specifiedVariableList != null) {
      this.specifiedVariableNodes = specifiedVariableList as VariableNode[];
    }
  }

  beforeBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    this.variableNodes = this.specifiedVariableNodes == null ?
        session_util.getVariableNodesFromEvaluationSet(runtime.nodes) :
        this.specifiedVariableNodes;
    if (batchSize !== this.prevBatchSize) {
      if (this.c != null) {
        this.c.dispose();
      }
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

  abstract afterBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap): void;

  dispose() {
    if (this.c != null) {
      this.c.dispose();
    }
    this.one.dispose();
  }

  protected variableGradients = new TensorArrayMap();
  protected prevBatchSize: number;
  protected one = Scalar.new(1);
  protected c: Scalar;
}
