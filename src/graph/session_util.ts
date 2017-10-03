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

// tslint:disable-next-line:max-line-length
import {InputProvider} from '../data/input_provider';
import {NDArrayMath} from '../math/math';
import {NDArray} from '../math/ndarray';
import * as util from '../util';

// tslint:disable-next-line:max-line-length
import {ConstantNode, Node, PlaceholderNode, Tensor, VariableNode} from './graph';
import * as graph_util from './graph_util';
import {Operation} from './ops/op';
import {FeedDictionary} from './session';
import {SummedTensorArrayMap, TensorArrayMap} from './tensor_array_map';

/**
 * Creates an array of graph nodes that stop traversal, based on the contents
 * of the provided FeedDictionary. This is a simple 1:1 extraction of nodes from
 * the FeedDictionary.
 *
 * @hidden
 * @param feedDictionary The FeedDictionary to scan for termination nodes.
 * @return an array of Nodes which halt traversal when visited.
 */
export function getTerminatingNodesFromFeedDictionary(
    feedDictionary: FeedDictionary): Node[] {
  return Object.keys(feedDictionary.dict)
      .map(tensorID => feedDictionary.dict[+tensorID].tensor.node);
}

/**
 * Given a tensor and a feed dictionary, computes the set of nodes that need to
 * be evaluated to perform inference.
 *
 * @hidden
 * @param evalTensors The list of tensors to eventually be evaluated.
 * @param feedDictionary The populated feed dictionary.
 * @return The set of nodes to evaluate, in evaluation order.
 */
export function getOrderedEvaluationSetFromEvalTensor(
    evalTensors: Tensor[], feedDictionary: FeedDictionary): Node[] {
  const terminatingNodes =
      getTerminatingNodesFromFeedDictionary(feedDictionary);
  const evalNodes = evalTensors.map(x => x.node);
  const unorderedEvaluationSet =
      graph_util.getUnorderedEvaluationSet(evalNodes, terminatingNodes);
  const orderedEvaluationSet =
      graph_util.getOrderedEvaluationSet(unorderedEvaluationSet);
  return orderedEvaluationSet;
}

/**
 * Traverses the provided node array and adds all persistent node NDArrays to
 * the provided TensorArrayMap.
 *
 * @hidden
 * @param evaluationSet The array of nodes to scan.
 * @param tensorArrayMap The map that receives the NDArrays from persistent
 * nodes.
 */
export function addPersistentArraysToTensorArrayMap(
    evaluationSet: Node[], tensorArrayMap: TensorArrayMap) {
  evaluationSet.forEach(node => {
    if (node instanceof VariableNode || node instanceof ConstantNode) {
      tensorArrayMap.set(node.output, node.data);
    }
  });
}

/**
 * @hidden
 */
export function getVariableNodesFromEvaluationSet(evaluationSet: Node[]):
    VariableNode[] {
  const nodes: VariableNode[] = [];
  evaluationSet.forEach(node => {
    if (node instanceof VariableNode) {
      nodes.push(node);
    }
  });
  return nodes;
}

/**
 * @hidden
 */
export function throwIfFeedDictionaryContainsNDArrays(
    feedDictionary: FeedDictionary) {
  Object.keys(feedDictionary.dict).forEach(tensorID => {
    if (feedDictionary.dict[+tensorID].data instanceof NDArray) {
      throw new Error(
          'training requires FeedDictionary entries to be InputProviders' +
          'and not NDArrays.');
    }
  });
}

/**
 * @hidden
 */
export function loadInputsFromFeedDictionaryToTensorArrayMap(
    batchFeed: FeedDictionary, activations: TensorArrayMap, math: NDArrayMath) {
  Object.keys(batchFeed.dict).forEach(tensorID => {
    const feedEntry = batchFeed.dict[+tensorID];

    let data: NDArray;
    if (feedEntry.data instanceof NDArray) {
      data = feedEntry.data as NDArray;
    } else {
      const provider = feedEntry.data as InputProvider;
      data = provider.getNextCopy(math);
    }

    util.assert(
        util.arraysEqual(feedEntry.tensor.shape, data.shape),
        `Error loading FeedEntry: feeding NDArray of shape ${data.shape} ` +
            `does not match Tensor (id: ${feedEntry.tensor.id}) shape: ` +
            `${feedEntry.tensor.shape}.`);
    activations.set(feedEntry.tensor, data);
  });
}

/**
 * @hidden
 */
export function releaseFeedDictionaryInputsFromTensorArrayMap(
    batchFeed: FeedDictionary, activations: TensorArrayMap, math: NDArrayMath) {
  Object.keys(batchFeed.dict).forEach(tensorID => {
    const feedEntry = batchFeed.dict[+tensorID];

    if (!(feedEntry.data instanceof NDArray)) {
      const provider = feedEntry.data as InputProvider;

      const feedEntryArray = activations.get(feedEntry.tensor);
      provider.disposeCopy(math, feedEntryArray);
    }

    activations.delete(feedEntry.tensor);
  });
}

/**
 * Removes all nodes from the provided Node array whose output tensors exist in
 * the provided feed dictionary. After calling this, the Node array should
 * contain zero Placeholder nodes, or the user has failed to provide a feed for
 * a Placeholder node.
 *
 * @hidden
 * @param feedDictionary The FeedDictionary to process.
 * @param evaluationSet The array of nodes to remove input nodes from.
 */
export function removeFeedDictionaryNodesFromEvaluationSet(
    feedDictionary: FeedDictionary, evaluationSet: Node[]) {
  let i = 0;
  while (i < evaluationSet.length) {
    const node = evaluationSet[i];
    if (feedDictionary.dict[node.output.id] != null) {
      evaluationSet.splice(i, 1);
    } else {
      ++i;
    }
  }
}

/**
 * Disposes any NDArrays on the tensorArrayMap from operation outputs and sets
 * the value to null.
 *
 * @hidden
 * @param evaluationSet The set of nodes to be evaluated.
 * @param tensorArrayMap The map to dispose and initialize.
 */
export function disposeAndInitializeOperationOutputs(
    evaluationSet: Node[], tensorArrayMap: TensorArrayMap) {
  evaluationSet.forEach(node => {
    if (!graph_util.isInputNode(node)) {
      if (!graph_util.isPassthroughNode(node, tensorArrayMap)) {
        tensorArrayMap.disposeArray(node.output);
      }
      tensorArrayMap.set(node.output, null);
    }
  });
}

/**
 * Disposes any NDArrays on the tensorArrayMap from derivatives of operation
 * inputs and sets the value to null.
 *
 * @hidden
 * @param evaluationSet The set of nodes to be evaluated.
 * @param gradients The gradient map to dispose and initialize.
 */
export function disposeAndInitializeOperationInputGradients(
    evaluationSet: Node[], gradients: SummedTensorArrayMap) {
  evaluationSet.forEach(node => {
    Object.keys(node.inputs).forEach(inputName => {
      const input = node.inputs[inputName];
      if (gradients.get(input, true) !== gradients.get(node.output, true)) {
        gradients.disposeArray(input);
      }
      gradients.nullify(input);
    });
  });
}

/**
 * Calls underlying operation disposeTransientArrays methods which clean up any
 * NDArrays which operations may have created during a run.
 *
 * @hidden
 * @param operationNodes The array of Nodes to traverse.
 * @param outputTensor The tensor being evaluated.
 * @param map The TensorArrayMap to operate on.
 */
export function disposeTransientOperationArrays(
    operations: Operation[], activations: TensorArrayMap,
    gradients: SummedTensorArrayMap) {
  operations.forEach(op => op.disposeTransientArrays(activations, gradients));
}

/**
 * Iterates the provided Node array and throws an exception if there are any
 * Placeholder nodes present. Call after the evaluation set has been pruned with
 * the accompanying FeedDictionary to ensure that all inputs have been resolved.
 *
 * @hidden
 * @param evaluationSet The array of nodes to scan.
 */
export function throwErrorIfEvaluationSetContainsPlaceholderNodes(
    evaluationSet: Node[]) {
  evaluationSet.forEach(node => {
    if (node instanceof PlaceholderNode) {
      const shape = '[' + node.output.shape.join(', ') + ']';
      throw new Error(
          'Placeholder node "' + node.name + '" ' + shape +
          ' not present in feed dictionary.');
    }
  });
}
