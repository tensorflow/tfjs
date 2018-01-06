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

import {InputProvider} from '../data/input_provider';
import {NDArrayMath} from '../math/math';
import {NDArray, Scalar} from '../math/ndarray';
import {Optimizer} from '../math/optimizers/optimizer';
import * as util from '../util';

import {Graph, Node, Tensor} from './graph';
import * as operation_emitter from './operation_emitter';
import {Operation} from './ops/op';
import * as session_util from './session_util';
import {SummedTensorArrayMap, TensorArrayMap} from './tensor_array_map';

/**
 * FeedEntry associates a tensor with user-provided NDArray data.
 */
export type FeedEntry = {
  tensor: Tensor,
  data: NDArray|InputProvider
};

/**
 * A FeedDictionary holds a map from tensors to user-provided NDArrays. Feed
 * dictionaries represent the 'entry points' of evaluation, since graph nodes
 * that are replaced by feeds don't need to have their input nodes evaluated.
 * Feed dictionaries usually provide NDArray data for Placeholder nodes, but any
 * node in the graph can be replaced by a feed dictionary entry.
 *
 * @hidden
 */
export class FeedDictionary {
  dict: {[tensorID: number]: FeedEntry} = {};

  /**
   * Optionally construct a FeedDictionary from an array of entries.
   * @param feedEntries Optional array of FeedEntry objects.
   */
  constructor(feedEntries?: FeedEntry[]) {
    if (feedEntries) {
      feedEntries.forEach(entry => this.dict[entry.tensor.id] = entry);
    }
  }
}

export enum CostReduction {
  NONE,
  SUM,
  MEAN
}

/**
 * A Session maintains the runtime state required to efficiently evaluate nodes.
 * On their own, graph objects are very lightweight logical topologies; they
 * have no relationship with the GPU. Sessions encapsulate the evaluation of
 * nodes, the management of GPU resources, the caching of evaluation paths, and
 * anything else required to evaluate or train a network.
 */
export class Session {
  /**
   * @param graph The graph to associate with this Session.
   * @param math The NDArrayMath interface that this Session should use.
   */
  constructor(graph: Graph, private math: NDArrayMath) {
    this.gradientArrayMap = new SummedTensorArrayMap(this.math);
  }

  /**
   * Release all system resources associated with this Session.
   */
  dispose() {
    this.activationArrayMap.dispose();
    Object.keys(this.runtimeCache).forEach(key => {
      const runtime = this.runtimeCache[key];
      if (runtime.operations) {
        runtime.operations.forEach(op => op.dispose());
      }
    });
    this.runtimeCache = {};
    if (this.batchSizeScalar != null) {
      this.batchSizeScalar.dispose();
    }
    this.oneScalar.dispose();
  }

  /**
   * Evaluate a list of tensors, using the provided feed entries to provide
   * upstream NDArray input.
   * When using a `NDArrayMath` object in safe mode this must be used in a
   * math.scope().
   * @param tensors The list of tensors to evaluate.
   * @param feedEntries List of `FeedEntry` to read when replacing graph
   * tensors with NDArrays.
   * @return The computed values of the tensors.
   */
  evalAll(tensors: Tensor[], feedEntries: FeedEntry[]): NDArray[] {
    return this.math.scope(() => {
      const feed = new FeedDictionary(feedEntries);
      const runtime = this.getOrCreateRuntime(tensors, feed);

      const activations = this.activationArrayMap;

      session_util.disposeAndInitializeOperationOutputs(
          runtime.nodes, activations);
      session_util.disposeTransientOperationArrays(
          runtime.operations, this.activationArrayMap, this.gradientArrayMap);

      session_util.addPersistentArraysToTensorArrayMap(
          runtime.nodes, activations);
      session_util.loadInputsFromFeedDictionaryToTensorArrayMap(
          feed, activations, this.math);

      runtime.operations.forEach(op => op.feedForward(this.math, activations));

      const results = tensors.map(x => activations.get(x));
      tensors.forEach(x => activations.delete(x));

      session_util.releaseFeedDictionaryInputsFromTensorArrayMap(
          feed, activations, this.math);

      return results;
    });
  }

  /**
   * Evaluate a tensor, using the provided feed entries to provide
   * upstream NDArray input.
   *
   * @param tensor The tensor to evaluate.
   * @param feedEntries List of `FeedEntry` to read when replacing graph
   * tensors with NDArrays.
   * @return The computed value of the tensor.
   */
  eval(tensor: Tensor, feedEntries: FeedEntry[]): NDArray {
    return this.evalAll([tensor], feedEntries)[0];
  }

  /**
   * Trains a batch.
   * Returns a reduced cost if the costReduction parameter is set.
   * When using a `NDArrayMath` object in safe mode this must be used in a
   * math.scope().
   * @param costTensor A tensor representing the cost to optimize. Should be a
   * scalar.
   * @param feedEntries Feed entries for this train run. Provides inputs.
   * @param batchSize Batch size for this train loop.
   * @param optimizer An optimizer to perform weight updates.
   * @param costReduction An option to allow the user to get a summed, averaged,
   * or no cost back.
   * @return The reduced cost, if cost reduction is not NONE. The user is
   * responsible for disposing the cost NDArray between train loops.
   */
  train(
      costTensor: Tensor, feedEntries: FeedEntry[], batchSize: number,
      optimizer: Optimizer, costReduction = CostReduction.NONE): Scalar {
    util.assert(
        util.isScalarShape(costTensor.shape),
        'Cost tensor for training must be a scalar value.');

    if (this.prevBatchSize !== batchSize) {
      this.prevBatchSize = batchSize;
      if (this.batchSizeScalar != null) {
        this.batchSizeScalar.dispose();
      }
      this.batchSizeScalar = this.math.keep(Scalar.new(batchSize));
    }

    const feed = new FeedDictionary(feedEntries);
    session_util.throwIfFeedDictionaryContainsNDArrays(feed);

    const runtime = this.getOrCreateRuntime([costTensor], feed);
    const inferenceOperations = runtime.operations;
    const backPropOperations = runtime.operations.slice().reverse();
    const activations = this.activationArrayMap;
    const gradients = this.gradientArrayMap;
    gradients.nullify(costTensor);
    gradients.add(costTensor, this.oneScalar);

    session_util.addPersistentArraysToTensorArrayMap(
        runtime.nodes, activations);

    optimizer.beforeBatch(
        this.math, batchSize, runtime, activations, gradients);

    return this.math.scope(() => {
      let cost = Scalar.new(0);

      for (let i = 0; i < batchSize; ++i) {
        session_util.disposeAndInitializeOperationOutputs(
            runtime.nodes, activations);
        session_util.disposeAndInitializeOperationInputGradients(
            runtime.nodes, gradients);
        session_util.disposeTransientOperationArrays(
            runtime.operations, activations, gradients);

        session_util.loadInputsFromFeedDictionaryToTensorArrayMap(
            feed, activations, this.math);

        inferenceOperations.forEach(
            op => op.feedForward(this.math, activations));
        backPropOperations.forEach(
            op => op.backProp(this.math, activations, gradients));

        optimizer.afterExample(this.math, runtime, activations, gradients);

        session_util.releaseFeedDictionaryInputsFromTensorArrayMap(
            feed, activations, this.math);

        cost = this.updateCostForExample(
            cost, activations.get(costTensor) as Scalar<'float32'>,
            costReduction);
      }

      optimizer.afterBatch(
          this.math, batchSize, runtime, activations, gradients);

      return this.updateCostForBatch(cost, costReduction);
    });
  }

  private updateCostForExample(
      totalCost: Scalar<'float32'>, currCost: Scalar<'float32'>,
      costReduction: CostReduction): Scalar<'float32'> {
    if (costReduction === CostReduction.MEAN ||
        costReduction === CostReduction.SUM) {
      return this.math.add(totalCost, currCost);
    }
    return totalCost;
  }

  private updateCostForBatch(
      totalCost: Scalar<'float32'>,
      costReduction: CostReduction): Scalar<'float32'> {
    if (costReduction === CostReduction.MEAN) {
      return this.math.divide(totalCost, this.batchSizeScalar);
    }
    return totalCost;
  }

  private getOrCreateRuntime(tensors: Tensor[], feed: FeedDictionary):
      SessionRuntime {
    const key = this.makeRuntimeCacheKey(tensors, feed);
    let runtime = this.runtimeCache[key];
    if (runtime === undefined) {
      const nodes =
          session_util.getOrderedEvaluationSetFromEvalTensor(tensors, feed);
      session_util.removeFeedDictionaryNodesFromEvaluationSet(feed, nodes);
      session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes(nodes);
      const operations = operation_emitter.emitFromGraphNodes(nodes);
      runtime = {nodes, operations};
      this.runtimeCache[key] = runtime;
    }

    return runtime;
  }

  private makeRuntimeCacheKey(tensors: Tensor[], feed: FeedDictionary): string {
    return tensors.map(x => x.id).sort().join('_') + '__' +
        Object.keys(feed.dict).sort().join('_');
  }

  /** Maps each output tensor of the graph to its activation value. */
  activationArrayMap = new TensorArrayMap();

  /** Maps each tensor of the graph to its derivative wrt the cost function. */
  gradientArrayMap: SummedTensorArrayMap;
  private runtimeCache: {[key: string]: SessionRuntime} = {};
  /** Batch size of the previous train() call. */
  private prevBatchSize: number;
  private batchSizeScalar: Scalar;
  private oneScalar = Scalar.new(1);
}

/** @hidden */
export type SessionRuntime = {
  nodes: Node[]; operations: Operation[];
};
