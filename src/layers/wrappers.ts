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
 * Layers that augment the functionality of a base layer.
 */

// tslint:disable:max-line-length
import {Tensor} from '@tensorflow/tfjs-core';

import * as K from '../backend/deeplearnjs_backend';
import {Layer, LayerConfig} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {Shape, TensorInterface} from '../types';
import {ConfigDict, LayerVariable, RegularizerFn, RnnStepFunction, SymbolicTensor} from '../types';
import * as generic_utils from '../utils/generic_utils';

import {RNN} from './recurrent';
import {deserialize} from './serialization';

// tslint:enable:max-line-length

export interface WrapperLayerConfig extends LayerConfig {
  /**
   * The layer to be wrapped.
   */
  layer: Layer;
}

/**
 * Abstract wrapper base class.
 *
 * Wrappers take another layer and augment it in various ways.
 * Do not use this class as a layer, it is only an abstract base class.
 * Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.
 */
export abstract class Wrapper extends Layer {
  readonly layer: Layer;
  /**
   * Tracks mapping of Wrapper inputs to inner layer inputs. Useful when
   * the inner layer has update ops that depend on its inputs (as opposed
   * to the inputs to the Wrapper layer).
   */
  private inputMap: {[key: string]: SymbolicTensor[]};

  constructor(config: WrapperLayerConfig) {
    // Porting Note: In PyKeras, `self.layer` is set prior to the calling
    //   `super()`. But we can't do that here due to TypeScript's restriction.
    //   See: https://github.com/Microsoft/TypeScript/issues/8277
    //   As a result, we have to add checks in `get trainable()` and
    //   `set trainable()` below in order to prevent using `this.layer` when
    //   its value is `undefined`. The super constructor does use the getter
    //   and the setter of `this.layer`.
    super(config);
    this.layer = config.layer;
    this.inputMap = {};
  }

  build(inputShape: Shape|Shape[]): void {
    this.built = true;
  }

  // TODO(cais): Implement activityRegularizer getter.

  get trainable(): boolean {
    // Porting Note: the check of `this.layer` here is necessary due to the
    //   way the `constructor` of this class is written (see Porting Note
    //   above).
    if (this.layer != null) {
      return this.layer.trainable;
    } else {
      return false;
    }
  }

  set trainable(value: boolean) {
    // Porting Note: the check of `this.layer` here is necessary due to the
    //   way the `constructor` of this class is written (see Porting Note
    //   above).
    if (this.layer != null) {
      this.layer.trainable = value;
    }
  }

  get trainableWeights(): LayerVariable[] {
    return this.layer.trainableWeights;
  }
  // TODO(cais): Implement setter for trainableWeights.

  get nonTrainableWeights(): LayerVariable[] {
    return this.layer.nonTrainableWeights;
  }
  // TODO(cais): Implement setter for nonTrainableWeights.

  get updates(): TensorInterface[] {
    // tslint:disable-next-line:no-any
    return (this.layer as any)._updates;
  }

  // TODO(cais): Implement getUpdatesFor().

  get losses(): RegularizerFn[] {
    return this.layer.losses;
  }

  // TODO(cais): Implement getLossesFor().

  getWeights(): Tensor[] {
    return this.layer.getWeights();
  }

  setWeights(weights: Tensor[]): void {
    this.layer.setWeights(weights);
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      'layer': {
        'className': this.layer.constructor.name,
        'config': this.layer.getConfig(),
      }
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  static fromConfig<T>(
      cls: generic_utils.Constructor<T>, config: ConfigDict,
      customObjects = {} as ConfigDict): T {
    const layerConfig = config['layer'] as ConfigDict;
    const layer = deserialize(layerConfig, customObjects) as Layer;
    delete config['layer'];
    const newConfig = {layer};
    Object.assign(newConfig, config);
    return new cls(newConfig);
  }
}

/**
 * This wrapper applies a layer to every temporal slice of an input.
 *
 * The input should be at least 3D,  and the dimension of the index `1` will be
 * considered to be the temporal dimension.
 *
 * Consider a batch of 32 samples, where each sample is a sequence of 10 vectors
 * of 16 dimensions. The batch input shape of the layer is then `[32,  10,
 * 16]`, and the `inputShape`, not including the sample dimension, is
 * `[10, 16]`.
 *
 * You can then use `TimeDistributed` to apply a `Dense` layer to each of the 10
 * timesteps, independently:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.timeDistributed({
 *   layer: tf.layers.dense({units: 8}),
 *   inputShape: [10, 16],
 * }));
 *
 * // Now model.outputShape = [null, 10, 8].
 * // The output will then have shape `[32, 10, 8]`.
 *
 * // In subsequent layers, there is no need for `inputShape`:
 * model.add(tf.layers.timeDistributed({layer: tf.layers.dense({units: 32})}));
 * // Now model.outputShape = [null, 10, 32].
 * ```
 *
 * The output will then have shape `[32, 10, 32]`.
 *
 * `TimeDistributed` can be used with arbitrary layers, not just `Dense`, for
 * instance a `Conv2D` layer.
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.timeDistributed({
 *   layer: tf.layers.conv2d({filters: 64, kernelSize: [3, 3]}),
 *   inputShape: [10, 299, 299, 3],
 * }));
 * ```
 */
export class TimeDistributed extends Wrapper {
  constructor(config: WrapperLayerConfig) {
    super(config);
    this.supportsMasking = true;
  }

  build(inputShape: Shape|Shape[]): void {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    if (inputShape.length < 3) {
      throw new ValueError(
          `TimeDistributed layer expects an input shape >= 3D, but received ` +
          `input shape ${JSON.stringify(inputShape)}`);
    }
    this.inputSpec = [{shape: inputShape}];
    const childInputShape = [inputShape[0]].concat(inputShape.slice(2));
    if (!this.layer.built) {
      this.layer.build(childInputShape);
      this.layer.built = true;
    }
    super.build(inputShape);
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    const childInputShape = [inputShape[0]].concat(inputShape.slice(2));
    const childOutputShape =
        this.layer.computeOutputShape(childInputShape) as Shape;
    const timesteps = inputShape[1];
    return [childOutputShape[0], timesteps].concat(childOutputShape.slice(1));
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    // TODO(cais): Add 'training' and 'useLearningPhase' to kwargs.
    inputs = generic_utils.getExactlyOneTensor(inputs);
    // Porting Note: In tfjs-layers, `inputs` are always concrete tensor values.
    // Hence the inputs can't have an undetermined first (batch) dimension,
    // which is why we always use the K.rnn approach here.
    const step: RnnStepFunction = (inputs: Tensor, states: Tensor[]) => {
      // TODO(cais): Add useLearningPhase.
      const output = this.layer.call(inputs, kwargs) as Tensor;
      return [output, []];
    };
    const rnnOutputs =
        K.rnn(step, inputs, [], false, null, null, false, inputs.shape[1]);
    const y = rnnOutputs[1];
    // TODO(cais): Add activity regularization.
    // TODO(cais): Add useLearningPhase.
    return y;
  }
}
generic_utils.ClassNameMap.register('TimeDistributed', TimeDistributed);

export enum BidirectionalMergeMode {
  SUM,
  MUL,
  CONCAT,
  AVE,
}
generic_utils.SerializableEnumRegistry.register('merge_mode', {
  'sum': BidirectionalMergeMode.SUM,
  'mul': BidirectionalMergeMode.MUL,
  'concat': BidirectionalMergeMode.CONCAT,
  'ave': BidirectionalMergeMode.AVE,
});

export interface BidirectionalLayerConfig extends WrapperLayerConfig {
  /**
   * The instance of an `RNN` layer to be wrapped.
   */
  layer: RNN;

  /**
   * Mode by which outputs of the forward and backward RNNs are combinied.
   * If `null` or `undefined`, the output will not be combined, they will be
   * returned as an `Array`.
   */
  mergeMode?: BidirectionalMergeMode;
}

export class Bidirectional extends Wrapper {
  private forwardLayer: RNN;
  private backwardLayer: RNN;
  private mergeMode: BidirectionalMergeMode;
  private returnSequences: boolean;
  private returnState: boolean;
  private _trainable: boolean;

  constructor(config: BidirectionalLayerConfig) {
    super(config);
    this.forwardLayer = config.layer;
    // TODO(cais): Perform shallow copy if necessary.
    const layerConfig = config.layer.getConfig();
    layerConfig['goBackwards'] =
        layerConfig['goBackwards'] === true ? false : true;
    this.backwardLayer = deserialize(
        {className: config.layer.constructor.name, config: layerConfig});
    this.forwardLayer.name = 'forward_' + this.forwardLayer.name;
    this.backwardLayer.name = 'backward_' + this.backwardLayer.name;
    this.mergeMode = config.mergeMode;
    if (config.weights) {
      throw new NotImplementedError(
          'weights support is not implemented for Bidirectional layer yet.');
    }
    this._stateful = config.layer.stateful;
    this.returnSequences = config.layer.returnSequences;
    this.returnState = config.layer.returnState;
    this.supportsMasking = true;
    this._trainable = true;
    this.inputSpec = config.layer.inputSpec;
  }

  get trainable(): boolean {
    return this._trainable;
  }

  set trainable(value: boolean) {
    // Porting Note: the check of `this.layer` here is necessary due to the
    //   way the `constructor` of this class is written (see Porting Note
    //   above).
    this._trainable = value;
    if (this.forwardLayer != null) {
      this.forwardLayer.trainable = value;
    }
    if (this.backwardLayer != null) {
      this.backwardLayer.trainable = value;
    }
  }

  getWeights(): Tensor[] {
    return this.forwardLayer.getWeights().concat(
        this.backwardLayer.getWeights());
  }

  setWeights(weights: Tensor[]): void {
    const numWeights = weights.length;
    const numeightsOver2 = Math.floor(numWeights / 2);
    this.forwardLayer.setWeights(weights.slice(0, numeightsOver2));
    this.backwardLayer.setWeights(weights.slice(numeightsOver2));
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    let layerShapes: Shape|Shape[] =
        this.forwardLayer.computeOutputShape(inputShape);
    if (!(Array.isArray(layerShapes) && Array.isArray(layerShapes[0]))) {
      layerShapes = [layerShapes as Shape];
    }
    layerShapes = layerShapes as Shape[];

    let outputShape: Shape;
    let outputShapes: Shape[];
    let stateShape: Shape[];
    if (this.returnState) {
      stateShape = layerShapes.slice(1);
      outputShape = layerShapes[0];
    } else {
      outputShape = layerShapes[0];
    }
    outputShape = outputShape as Shape;
    if (this.mergeMode === BidirectionalMergeMode.CONCAT) {
      outputShape[outputShape.length - 1] *= 2;
      outputShapes = [outputShape];
    } else if (this.mergeMode == null) {
      outputShapes = [outputShape, outputShape.slice()];
    } else {
      outputShapes = [outputShape];
    }

    if (this.returnState) {
      if (this.mergeMode == null) {
        return outputShapes.concat(stateShape).concat(stateShape.slice());
      }
      return [outputShape].concat(stateShape).concat(stateShape.slice());
    }
    return generic_utils.singletonOrArray(outputShapes);
  }

  apply(
      inputs: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
      // tslint:disable-next-line:no-any
      kwargs?: any): Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[] {
    let initialState: Tensor[]|SymbolicTensor[] = null;
    if (kwargs != null) {
      initialState = kwargs['initialState'];
    }
    if (Array.isArray(inputs)) {
      initialState = (inputs as Tensor[] | SymbolicTensor[]).slice(1);
      inputs = (inputs as Tensor[] | SymbolicTensor[])[0];
    }

    if (initialState == null || initialState.length === 0) {
      const applyOutputs = super.apply(inputs, kwargs);
      return applyOutputs;
    } else {
      throw new NotImplementedError(
          'The support for initial states is not implemented for ' +
          'Bidirectional layers yet.');
    }
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    if (kwargs['mask'] != null) {
      throw new NotImplementedError(
          'The support for masking is not implemented for ' +
          'Bidirectional layers yet.');
    }
    if (kwargs['initialState'] != null) {
      throw new NotImplementedError(
          'The support for initial states is not implemented for ' +
          'Bidirectional layers yet.');
    }

    // TODO(cais): Implement support for initial state.
    let y = this.forwardLayer.call(inputs, kwargs);
    let yRev = this.backwardLayer.call(inputs, kwargs);

    let states: Tensor[];
    if (this.returnState) {
      if (Array.isArray(y)) {
        states = (y as Tensor[]).slice(1).concat((yRev as Tensor[]).slice(1));
      } else {
      }
      y = (y as Tensor[])[0];
      yRev = (yRev as Tensor[])[0];
    }

    if (this.returnSequences) {
      yRev = K.reverse(yRev as Tensor, 1);
    }

    let output: Tensor|Tensor[];
    if (this.mergeMode === BidirectionalMergeMode.CONCAT) {
      output = K.concatenate([y as Tensor, yRev as Tensor]);
    } else if (this.mergeMode === BidirectionalMergeMode.SUM) {
      output = K.add(y as Tensor, yRev as Tensor);
    } else if (this.mergeMode === BidirectionalMergeMode.AVE) {
      output = K.scalarTimesArray(
          K.getScalar(0.5), K.add(y as Tensor, yRev as Tensor));
    } else if (this.mergeMode === BidirectionalMergeMode.MUL) {
      output = K.multiply(y as Tensor, yRev as Tensor);
    } else if (this.mergeMode == null) {
      output = [y as Tensor, yRev as Tensor];
    }

    // TODO(cais): Properly set learning phase.
    if (this.returnState) {
      if (this.mergeMode == null) {
        return (output as Tensor[]).concat(states);
      }
      return [output as Tensor].concat(states);
    }
    return output;
  }

  resetStates(states?: Tensor|Tensor[]): void {
    this.forwardLayer.resetStates();
    this.backwardLayer.resetStates();
  }

  build(inputShape: Shape|Shape[]): void {
    K.nameScope(this.forwardLayer.name, () => {
      this.forwardLayer.build(inputShape);
    });
    K.nameScope(this.backwardLayer.name, () => {
      this.backwardLayer.build(inputShape);
    });
    this.built = true;
  }

  // TODO(cais): Implement computeMask().

  get trainableWeights(): LayerVariable[] {
    return this.forwardLayer.trainableWeights.concat(
        this.backwardLayer.trainableWeights);
  }

  get nonTrainableWeights(): LayerVariable[] {
    return this.forwardLayer.nonTrainableWeights.concat(
        this.backwardLayer.nonTrainableWeights);
  }

  // TODO(cais): Implement constraints().
  // TODO(cais): Implement getConfig().
}
generic_utils.ClassNameMap.register('Bidirectional', Bidirectional);
