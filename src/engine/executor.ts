/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Executor: Evaluates SymbolicTensor based on feeds.
 */

import {Tensor} from '@tensorflow/tfjs-core';
import {ValueError} from '../errors';
import {SymbolicTensor} from '../types';
import {InputLayer} from './topology';

/**
 * Helper function to check the dtype and shape compatibility of a feed value.
 */
function assertFeedCompatibility(key: SymbolicTensor, val: Tensor) {
  if (key.dtype != null && key.dtype !== val.dtype) {
    throw new ValueError(
        `The dtype of the feed (${val.dtype}) is incompatible with that of ` +
        `the key '${key.name}' (${key.dtype}).`);
  }

  if (key.shape != null) {
    if (key.shape.length !== val.shape.length) {
      throw new ValueError(
          `The rank of feed (${val.shape.length}) does not match the rank of ` +
          `the key (${key.shape.length}).`);
    }

    for (let i = 0; i < key.shape.length; ++i) {
      if (key.shape[i] != null && key.shape[i] !== val.shape[i]) {
        throw new ValueError(
            `The ${i}-th dimension of the feed (${val.shape[i]}) is ` +
            `incompatible with that of the key (${key.shape[i]}).`);
      }
    }
  }
}

/**
 * A concrete Tensor value for a symbolic tensor as the key.
 */
export interface Feed {
  key: SymbolicTensor;
  value: Tensor;
}

/**
 * FeedDict: A mapping from unique SymbolicTensors to feed values for them.
 * A feed value is a concrete value represented as an `Tensor`.
 */
export class FeedDict {
  private id2Value: {[id: number]: Tensor} = {};

  /**
   * Constructor, optionally does copy-construction.
   * @param feeds An Array of `Feed`s, or another `FeedDict`, in which case
   *   copy-construction will be performed.
   */
  constructor(feeds?: Feed[]|FeedDict) {
    if (feeds instanceof FeedDict) {
      for (const id in feeds.id2Value) {
        this.id2Value[id] = feeds.id2Value[id];
      }
    } else {
      if (feeds == null) {
        return;
      }
      for (const feed of feeds) {
        this.add(feed.key, feed.value);
      }
    }
  }

  /**
   * Add a key-value pair to the FeedDict.
   * @param key The key of the feed.
   * @param value The value of the feed.
   * @returns This `FeedDict`.
   * @throws ValueError: If the key `SymbolicTensor` already exists in the
   *   `FeedDict`.
   */
  add(key: SymbolicTensor, value: Tensor): FeedDict {
    assertFeedCompatibility(key, value);
    if (this.id2Value[key.id] == null) {
      this.id2Value[key.id] = value;
    } else {
      throw new ValueError(`Duplicate key: name=${key.name}, id=${key.id}`);
    }
    return this;
  }

  /**
   * Add a Feed to the FeedDict.
   * @param feed The new `Feed` to add.
   * @returns This `FeedDict`.
   */
  addFeed(feed: Feed) {
    this.add(feed.key, feed.value);
  }

  /**
   * Probe whether a key already exists in the FeedDict.
   * @param key
   */
  hasKey(key: SymbolicTensor): boolean {
    return this.id2Value[key.id] != null;
  }

  /**
   * Get the feed value for given key.
   * @param key
   * @returns If `key` exists, the corresponding feed value.
   * @throws ValueError: If `key` does not exist in this `FeedDict`.
   */
  getValue(key: SymbolicTensor): Tensor {
    if (this.id2Value[key.id] == null) {
      throw new ValueError(`Nonexistent key: ${JSON.stringify(key)}`);
    } else {
      return this.id2Value[key.id];
    }
  }
}

/**
 * Execute a SymbolicTensor by using concrete feed values.
 *
 * A `SymbolicTensor` object is a node in a computation graph of TF.js
 * Layers. The object is backed by a source layer and input
 * `SymbolicTensor`s to the source layer. This method evaluates
 * the `call()` method of the source layer, using concrete values of the inputs
 * obtained from either
 * * `feedDict`, if the input key exists in `feedDict`, or else,
 * * a recursive call to `execute()` itself.
 *
 * @param x: The `SymbolicTensor` to execute.
 * @param feedDict: The feed values, as base condition of the recursion.
 *   execution.
 * @param kwargs: Optional keyword arguments.
 * @returns Result of the execution.
 * @throws ValueError: If any `SymbolicTensor`s from `InputLayer`s
 *   encountered during the execution lacks a feed value in `feedDict`.
 */
export function execute(
    fetches: SymbolicTensor|SymbolicTensor[], feedDict: FeedDict,
    // tslint:disable-next-line:no-any
    kwargs?: any): Tensor|Tensor[]|[Tensor | Tensor[]] {
  const arrayFetches = Array.isArray(fetches);
  const fetchArray: SymbolicTensor[] =
      arrayFetches ? fetches as SymbolicTensor[] : [fetches as SymbolicTensor];

  const outputs: Tensor[] = [];
  const internalFeedDict = new FeedDict(feedDict);

  for (const fetch of fetchArray) {
    outputs.push(executeInternal(fetch, internalFeedDict, kwargs) as Tensor);
  }
  return arrayFetches ? outputs : outputs[0];
}

function executeInternal(
    fetch: SymbolicTensor, internalFeedDict: FeedDict,
    // tslint:disable-next-line:no-any
    kwargs?: any): Tensor {
  if (internalFeedDict.hasKey(fetch)) {
    return internalFeedDict.getValue(fetch);
  }
  if (fetch.sourceLayer instanceof InputLayer) {
    throw new ValueError(
        `Missing a feed value for SymbolicTensor from InputLayer ` +
        `'${InputLayer.name}'`);
  }

  const inputs = fetch.inputs;
  const inputValues: Tensor[] = [];
  for (const input of inputs) {
    // Recursive call.
    const inputVal = executeInternal(input, internalFeedDict, kwargs) as Tensor;
    inputValues.push(inputVal);
  }

  let output =
      fetch.sourceLayer.apply(inputValues, kwargs) as Tensor | Tensor[];
  if (!Array.isArray(output)) {
    output = [output];
  }
  const layerOutputs = getNodeOutputs(fetch);
  const outputSymbolicTensors =
      Array.isArray(layerOutputs) ? layerOutputs : [layerOutputs];
  for (let i = 0; i < outputSymbolicTensors.length; ++i) {
    internalFeedDict.add(outputSymbolicTensors[i], output[i]);
  }
  return output.length === 1 ? output[0] : output[fetch.outputTensorIndex];
}

/**
 * Get the symbolic output tensors of the node to which a given fetch belongs.
 * @param fetch The fetched symbolic tensor.
 * @returns The Array of symbolic tensors output by the node to which `fetch`
 *   belongs.
 */
function getNodeOutputs(fetch: SymbolicTensor): SymbolicTensor|
    SymbolicTensor[] {
  let layerOutputs: SymbolicTensor|SymbolicTensor[];
  if (fetch.sourceLayer.inboundNodes.length === 1) {
    layerOutputs = fetch.sourceLayer.output;
  } else {
    let nodeIndex: number = null;
    for (let i = 0; i < fetch.sourceLayer.inboundNodes.length; ++i) {
      for (const outputTensor of fetch.sourceLayer.inboundNodes[i]
               .outputTensors) {
        if (outputTensor.id === fetch.id) {
          nodeIndex = i;
          break;
        }
      }
    }
    layerOutputs = fetch.sourceLayer.getOutputAt(nodeIndex);
  }
  return layerOutputs;
}
