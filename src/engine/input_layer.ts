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
import {Kwargs, Shape} from '../types';

import {Layer, Node, SymbolicTensor, DisposeResult} from './topology';

/**
 * Constructor arguments for InputLayer.
 *
 * Note: You should provide only inputShape or batchInputShape (not both).
 * If only inputShape is provided, then the batchInputShape is determined by
 * the batchSize argument and the inputShape: [batchSize].concat(inputShape).
 */
export interface InputLayerConfig {
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

/**
 * An input layer is an entry point into a `Model`.
 *
 * `InputLayer` is generated automatically for `Sequential` models by specifying
 * the `inputshape` or `batchInputShape` for the first layer.  It should not be
 * specified explicitly.
 *
 * ```js
 * // Define a model which simply adds two inputs.
 * const inputA = tf.input({shape: [3]});
 * const inputB = tf.input({shape: [3]});
 * const sum = tf.layers.add().apply([inputA, inputB]);
 * const model = tf.model({inputs: [inputA, inputB], outputs: sum});
 * const batchSize = 2;
 * model.predict([tf.ones([batchSize, 3]), tf.ones([batchSize, 3])]).print();
 * ```
 */
export class InputLayer extends Layer {
  static readonly className = 'InputLayer';
  sparse: boolean;
  constructor(config: InputLayerConfig) {
    super({
      dtype: config.dtype,
      name: config.name != null ? config.name : getUid('input').toString()
    });
    // Normalize config.batchSize and config.sparse
    if (config.batchSize == null) {
      config.batchSize = null;
    }
    if (config.sparse == null) {
      config.sparse = false;
    }

    this.trainable = false;
    this.built = true;
    this.sparse = config.sparse;

    if (config.inputShape != null && config.batchInputShape != null) {
      throw new ValueError(
          'Only provide the inputShape OR ' +
          'batchInputShape argument to inputLayer, not both at the same time.');
    }
    let batchInputShape = config.batchInputShape;
    if (batchInputShape == null) {
      if (config.inputShape == null) {
        throw new ValueError(
            'An InputLayer should be passed either a ' +
            '`batchInputShape` or an `inputShape`.');
      } else {
        batchInputShape = [config.batchSize].concat(config.inputShape);
      }
    } else {
      // TODO(michaelterry): Backport to PyKeras
      if (config.batchSize != null) {
        throw new ValueError(
            'Cannot specify batchSize if batchInputShape is' +
            'specified when creating an InputLayer.');
      }
    }

    const dtype = config.dtype || 'float32';

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
    return {
      refCountAfterDispose: this._refCount,
      numDisposedVariables: 0
    };
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
serialization.SerializationMap.register(InputLayer);

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

/**
 * Used to instantiate an input to a model as a `SymbolicTensor`.
 *
 * Users should call the `input` factory function for
 * consistency with other generator functions.
 *
 * Example:
 *
 * ```js
 * // Defines a simple logistic regression model with 32 dimensional input
 * // and 3 dimensional output.
 * const x = tf.input({shape: [32]});
 * const y = tf.layers.dense({units: 3, activation: 'softmax'}).apply(x);
 * const model = tf.model({inputs: x, outputs: y});
 * model.predict(tf.ones([2, 32])).print();
 * ```
 *
 * Note: `input` is only necessary when using `model`. When using
 * `sequential`, specify `inputShape` for the first layer or use `inputLayer`
 * as the first layer.
 */
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
