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
import {Node, VariableNode} from '../../graph/graph';
import {SessionRuntime} from '../../graph/session';
import * as session_util from '../../graph/session_util';
// tslint:disable-next-line:max-line-length
import {SummedTensorArrayMap, TensorArrayMap} from '../../graph/tensor_array_map';
import {NDArrayMath} from '../../math/math';
import {DataType, NDArray, Scalar} from '../../math/ndarray';
import {NamedVariableMap} from '../../util';

export abstract class Optimizer {
  protected variableNodes: VariableNode[];
  protected specifiedVariableNodes: VariableNode[]|null;

  constructor(protected learningRate: number, specifiedVariableList?: Node[]) {
    if (specifiedVariableList != null) {
      this.specifiedVariableNodes = specifiedVariableList as VariableNode[];
    }
    this.one = ENV.math.keep(Scalar.new(1));
  }

  /**
   * Eager mode methods.
   */
  minimize<D extends DataType>(f: () => Scalar<D>, returnCost = false):
      Scalar<D>|null {
    const {value, gradients} = this.computeGradients(f);

    this.applyGradients(gradients);

    // Dispose gradients.
    const varNames = Object.keys(gradients);
    varNames.forEach(varName => gradients[varName].dispose());

    if (returnCost) {
      return value as Scalar<D>;
    } else {
      value.dispose();
      return null;
    }
  }

  computeGradients<D extends DataType>(f: () => Scalar<D>):
      {value: Scalar<D>, gradients: NamedVariableMap} {
    return ENV.math.variableGradients(f);
  }

  abstract applyGradients(variableGradients: NamedVariableMap): void;

  /**
   * Graph mode methods.
   */
  beforeBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap) {
    this.variableNodes = this.specifiedVariableNodes == null ?
        session_util.getVariableNodesFromEvaluationSet(runtime.nodes) :
        this.specifiedVariableNodes;
    if (batchSize !== this.prevBatchSize) {
      if (this.cGraph != null) {
        this.cGraph.dispose();
      }
      this.prevBatchSize = batchSize;
      this.cGraph = math.keep(Scalar.new(-this.learningRate / batchSize));
    }
    this.variableNodes.forEach(
        node => this.variableGradients.set(
            node.output, math.keep(NDArray.zeros(node.output.shape))));
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
    if (this.cGraph != null) {
      this.cGraph.dispose();
    }
    this.one.dispose();
    if (this.variableNodes != null) {
      this.variableNodes.forEach(node => {
        node.data.dispose();
      });
    }
    if (this.specifiedVariableNodes != null) {
      this.specifiedVariableNodes.forEach(node => {
        node.data.dispose();
      });
    }
  }

  protected variableGradients = new TensorArrayMap();
  protected prevBatchSize: number;
  protected one: Scalar;
  protected cGraph: Scalar;
}
