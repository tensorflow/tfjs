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

import {ENV} from '../../environment';
import {keep, tidy} from '../../globals';
import {Node} from '../../graph/graph';
import {SessionRuntime} from '../../graph/session';
import * as session_util from '../../graph/session_util';
// tslint:disable-next-line:max-line-length
import {SummedTensorArrayMap, TensorArrayMap} from '../../graph/tensor_array_map';
import {NDArrayMath} from '../../math/math';
import {scalar, zerosLike} from '../../math/ops';
import {Optimizer} from '../../math/optimizers/optimizer';
import {Scalar, Tensor} from '../../math/tensor';
import {NamedVariableMap} from '../../math/types';
import {variable} from '../tensor';

export class RMSPropOptimizer extends Optimizer {
  private c: Scalar;
  private epsilon: Scalar;
  private decay: Scalar;
  private momentum: Scalar;
  private oneMinusDecay: Scalar;

  private accumulatedMeanSquares: NamedVariableMap = {};
  private accumulatedMoments: NamedVariableMap = {};

  constructor(
      protected learningRate: number, decay = 0.9, momentum = 0.0,
      /** @deprecated only for graph */
      specifiedVariableList?: Node[], epsilon = 1e-8) {
    super(learningRate, specifiedVariableList);

    this.c = keep(scalar(learningRate));
    this.epsilon = keep(scalar(epsilon));
    this.decay = keep(scalar(decay));
    this.momentum = keep(scalar(momentum));
    this.oneMinusDecay = keep(scalar(1 - decay));
  }

  applyGradients(variableGradients: NamedVariableMap) {
    for (const variableName in variableGradients) {
      const value = ENV.engine.registeredVariables[variableName];
      if (this.accumulatedMeanSquares[variableName] == null) {
        const trainable = false;
        this.accumulatedMeanSquares[variableName] =
            variable(zerosLike(value), trainable);
      }
      if (this.accumulatedMoments[variableName] == null) {
        const trainable = false;
        this.accumulatedMoments[variableName] =
            variable(zerosLike(value), trainable);
      }

      const accumulatedMeanSquare = this.accumulatedMeanSquares[variableName];
      const accumulatedMoments = this.accumulatedMoments[variableName];
      const gradient = variableGradients[variableName];

      tidy(() => {
        const newAccumulatedMeanSquare =
            this.decay.mul(accumulatedMeanSquare)
                .add(this.oneMinusDecay.mul(gradient.square()));

        const newAccumulatedMoments =
            this.momentum.mul(accumulatedMoments)
                .add(this.c.mul(gradient).div(
                    newAccumulatedMeanSquare.add(this.epsilon).sqrt()));

        this.accumulatedMeanSquares[variableName].assign(
            newAccumulatedMeanSquare);
        this.accumulatedMoments[variableName].assign(newAccumulatedMoments);

        const newValue = value.sub(newAccumulatedMoments);
        value.assign(newValue);
      });
    }
  }

  // Graph
  /** @deprecated only for graph */
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
      this.cGraph = math.keep(scalar(this.learningRate / batchSize));
    }
    this.variableNodes.forEach(
        node => this.variableGradients.set(
            node.output, math.keep(Tensor.zeros(node.output.shape))));
    if (this.accumulatedMeanSquaredGraph.size() === 0) {
      this.variableNodes.forEach(node => {
        this.accumulatedMeanSquaredGraph.set(
            node.output, Tensor.zeros(node.output.shape));
        this.accumulatedMomentGraph.set(
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
        const oldMeanSquare = this.accumulatedMeanSquaredGraph.get(node.output);
        const oldMoment = this.accumulatedMomentGraph.get(node.output);

        // mean_square = decay * mean_square{t-1} +
        //          (1-decay) * gradient.square()
        // moment = momentum * mom{t - 1} +
        //          learning_rate * gradient / sqrt(mean_square + epsilon)
        // variable = variable - moment
        const meanSquare = math.scaledArrayAdd(
            this.decay, oldMeanSquare, this.oneMinusDecay, gradient.square());
        const moment = math.scaledArrayAdd(
            this.momentum, oldMoment, this.cGraph,
            gradient.div(meanSquare.add(this.epsilon).sqrt()));
        const variable = oldVariable.sub(moment);

        this.accumulatedMeanSquaredGraph.set(node.output, keep(meanSquare));
        this.accumulatedMomentGraph.set(node.output, keep(moment));
        activationArrayMap.set(node.output, keep(variable));

        node.data = variable;

        oldVariable.dispose();
        oldMeanSquare.dispose();
        oldMoment.dispose();
      });
    });

    this.variableGradients.dispose();
    this.variableGradients = new TensorArrayMap();
  }

  dispose() {
    super.dispose();
    this.c.dispose();
    this.epsilon.dispose();
    this.decay.dispose();
    this.momentum.dispose();
    this.oneMinusDecay.dispose();
    if (this.accumulatedMeanSquaredGraph != null) {
      this.accumulatedMeanSquaredGraph.dispose();
    }
    if (this.accumulatedMomentGraph != null) {
      this.accumulatedMomentGraph.dispose();
    }
    if (this.accumulatedMeanSquares != null) {
      Object.keys(this.accumulatedMeanSquares)
          .forEach(name => this.accumulatedMeanSquares[name].dispose());
    }
    if (this.accumulatedMoments != null) {
      Object.keys(this.accumulatedMoments)
          .forEach(name => this.accumulatedMoments[name].dispose());
    }
  }

  private accumulatedMeanSquaredGraph = new TensorArrayMap();
  private accumulatedMomentGraph = new TensorArrayMap();
}
