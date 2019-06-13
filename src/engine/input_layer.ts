/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataType, serialization, Tensor} from '@tensorflow/tfjs-core';

import {getUid} from '../backend/state';
import {ValueError} from '../errors';
import {Shape} from '../keras_format/common';
import {Kwargs} from '../types';

import {DisposeResult, Layer, Node, SymbolicTensor} from './topology';

/**
 * Constructor arguments for InputLayer.
 *
 * Note: You should provide only inputShape or batchInputShape (not both).
 * If only inputShape is provided, then the batchInputShape is determined by
 * the batchSize argument and the inputShape: [batchSize].concat(inputShape).
 */
export declare interface InputLayerArgs {
  /** Input shape, not including the batch axis. */
  inputShape?: Shape;
  /** Optional input batch size (integer or null). */
  batchSize?: number;
  /** Batch input shape, including the batch axis. */
  batchInputShape?: Shape;
  /** Datatype of the input.  */
  dtype?: DataType;
  /**
   * Whether the placeholder created is meant to be sparse.
   */
  sparse?: boolean;  // TODO(michaelterry): Not clear whether we'll need this.

  /** Name of the layer. */
  name?: string;
}

export class InputLayer extends Layer {
  /** @nocollapse */
  static readonly className = 'InputLayer';
  sparse: boolean;
  constructor(args: InputLayerArgs) {
    super({
      dtype: args.dtype,
      name: args.name != null ? args.name : getUid('input').toString()
    });
    // Normalize config.batchSize and config.sparse
    if (args.batchSize == null) {
      args.batchSize = null;
    }
    if (args.sparse == null) {
      args.sparse = false;
    }

    this.trainable = false;
    this.built = true;
    this.sparse = args.sparse;

    if (args.inputShape != null && args.batchInputShape != null) {
      throw new ValueError(
          'Only provide the inputShape OR ' +
          'batchInputShape argument to inputLayer, not both at the same time.');
    }
    let batchInputShape = args.batchInputShape;
    if (batchInputShape == null) {
      if (args.inputShape == null) {
        throw new ValueError(
            'An InputLayer should be passed either a ' +
            '`batchInputShape` or an `inputShape`.');
      } else {
        batchInputShape = [args.batchSize].concat(args.inputShape);
      }
    } else {
      // TODO(michaelterry): Backport to PyKeras
      if (args.batchSize != null) {
        throw new ValueError(
            'Cannot specify batchSize if batchInputShape is ' +
            'specified when creating an InputLayer.');
      }
    }

    const dtype = args.dtype || 'float32';

    this.batchInputShape = batchInputShape;
    this.dtype = dtype;
    // TODO(michaelterry): Backport this to PyKeras?
    this.inputSpec = [{shape: batchInputShape}];

    const inputTensor = new SymbolicTensor(
        this.dtype, this.batchInputShape, this, [], {}, this.name);
    inputTensor.nodeIndex = 0;
    inputTensor.tensorIndex = 0;

    // Create an input node to add to this.outboundNode.
    // (This call has side effects.)
    // tslint:disable-next-line:no-unused-expression
    new Node({
      outboundLayer: this,
      inboundLayers: [],
      nodeIndices: [],
      tensorIndices: [],
      inputTensors: [inputTensor],
      outputTensors: [inputTensor],
      inputMasks: [null],
      outputMasks: [null],
      inputShapes: [batchInputShape],
      outputShapes: [batchInputShape]
    });
  }

  apply(
      inputs: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
      kwargs?: Kwargs): Tensor|Tensor[]|SymbolicTensor {
    throw new ValueError(
        'Cannot pass any input to an ' +
        `InputLayer's apply() method. InputLayer name: ${this.name}`);
  }

  dispose(): DisposeResult {
    // dispose() for InputLayer is overridden as no-op.
    return {refCountAfterDispose: this._refCount, numDisposedVariables: 0};
  }

  getConfig(): serialization.ConfigDict {
    return {
      batchInputShape: this.batchInputShape,
      dtype: this.dtype,
      sparse: this.sparse,
      name: this.name
    };
  }
}
serialization.registerClass(InputLayer);

/**
 * Config for the Input function.
 *
 * Note: You should provide only shape or batchShape (not both).
 * If only shape is provided, then the batchShape becomes
 * [null].concat(inputShape).
 */
export interface InputConfig {
  /**
   * A shape, not including the batch size. For instance, `shape=[32]`
   * indicates that the expected input will be batches of 32-dimensional
   * vectors.
   */
  shape?: Shape;
  /**
   * A shape tuple (integer), including the batch size. For instance,
   * `batchShape=[10, 32]` indicates that the expected input will be batches of
   * 10 32-dimensional vectors. `batchShape=[null, 32]` indicates batches of an
   * arbitrary number of 32-dimensional vectors.
   */
  batchShape?: Shape;
  /**
   * An optional name string for the layer. Should be unique in a model (do not
   * reuse the same name twice). It will be autogenerated if it isn't provided.
   */
  name?: string;
  dtype?: DataType;
  /**
   * A boolean specifying whether the placeholder to be created is sparse.
   */
  sparse?: boolean;
}

export function Input(config: InputConfig): SymbolicTensor {
  if (config.batchShape == null && config.shape == null) {
    throw new Error(
        'Please provide to Input either a `shape`' +
        ' or a `batchShape` argument. Note that ' +
        '`shape` does not include the batch ' +
        'dimension.');
  }
  if (config.batchShape != null && config.shape != null) {
    // TODO(michaelterry): Backport to PyKeras.
    throw new ValueError(
        'Please provide either a `shape` or `batchShape` ' +
        'argument to Input, but not both.');
  }
  let batchShape = config.batchShape;
  if (config.shape != null && batchShape == null) {
    batchShape = [null].concat(config.shape);
  }

  let dtype = config.dtype;
  if (dtype == null) {
    dtype = 'float32';
  }

  const inputLayer = new InputLayer({
    batchInputShape: batchShape,
    name: config.name,
    dtype,
    sparse: config.sparse
  });

  const outputs = inputLayer.inboundNodes[0].outputTensors;
  return outputs[0];
}
