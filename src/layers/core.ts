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
 * TensorFlow.js Layers: Basic Layers.
 */

import {Scalar, serialization, Tensor, util} from '@tensorflow/tfjs-core';

// tslint:disable:max-line-length
import {ActivationFn, ActivationIdentifier, getActivation, serializeActivation} from '../activations';
import * as K from '../backend/tfjs_backend';
import {Constraint, ConstraintIdentifier, getConstraint, serializeConstraint} from '../constraints';
import {Layer, LayerConfig} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, serializeInitializer} from '../initializers';
import {getRegularizer, Regularizer, RegularizerIdentifier, serializeRegularizer} from '../regularizers';
import {Kwargs, Shape} from '../types';
import * as generic_utils from '../utils/generic_utils';
import {getExactlyOneTensor} from '../utils/generic_utils';
import * as math_utils from '../utils/math_utils';
import {LayerVariable} from '../variables';

// tslint:enable:max-line-length

export interface DropoutLayerConfig extends LayerConfig {
  /** Float between 0 and 1. Fraction of the input units to drop. */
  rate: number;

  /**
   * Integer array representing the shape of the binary dropout mask that will
   * be multiplied with the input.
   *
   * For instance, if your inputs have shape `(batchSize, timesteps, features)`
   * and you want the dropout mask to be the same for all timesteps, you can use
   * `noise_shape=(batch_size, 1, features)`.
   */
  noiseShape?: number[];

  /** An integer to use as random seed. */
  seed?: number;
}

/**
 * Applies
 * [dropout](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) to
 * the input.
 *
 * Dropout consists in randomly setting a fraction `rate` of input units to 0 at
 * each update during training time, which helps prevent overfitting.
 */
export class Dropout extends Layer {
  static className = 'Dropout';
  private readonly rate: number;
  private readonly rateScalar: Scalar;
  private readonly noiseShape: number[];
  private readonly seed: number;

  constructor(config: DropoutLayerConfig) {
    super(config);
    this.rate = Math.max(Math.min(config.rate, 1), 0);
    this.rateScalar = K.getScalar(this.rate);
    // So that the scalar doesn't get tidied up between executions.
    this.noiseShape = config.noiseShape;
    this.seed = config.seed;
    if (this.seed != null) {
      throw new NotImplementedError(
          'Non-default seed is not implemented in Dropout layer yet: ' +
          this.seed);
    }
    this.supportsMasking = true;
  }

  private getNoiseShape(input: Tensor): Shape {
    if (this.noiseShape == null) {
      return this.noiseShape;
    }
    const inputShape = input.shape;
    const noiseShape: Shape = [];
    for (let i = 0; i < this.noiseShape.length; ++i) {
      noiseShape.push(
          this.noiseShape[i] == null ? inputShape[i] : this.noiseShape[i]);
    }
    return noiseShape;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    this.invokeCallHook(inputs, kwargs);
    const input = generic_utils.getExactlyOneTensor(inputs);
    if (this.noiseShape != null &&
        !util.arraysEqual(input.shape, this.noiseShape)) {
      throw new NotImplementedError(
          'Non-default noise shape is not implemented in Dropout layer yet: ' +
          JSON.stringify(this.noiseShape));
    }
    if (0 < this.rate && this.rate < 1) {
      const training = kwargs['training'] == null ? false : kwargs['training'];
      const noiseShape = this.getNoiseShape(input);
      const output =
          K.inTrainPhase(
              () => K.dropout(input, this.rateScalar, noiseShape, this.seed),
              () => input, training) as Tensor;
      return output;
    }
    return inputs;
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      rate: this.rate,
      noiseShape: this.noiseShape,
      seed: this.seed,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.SerializationMap.register(Dropout);

export interface DenseLayerConfig extends LayerConfig {
  /** Positive integer, dimensionality of the output space. */
  units: number;
  /**
   * Activation function to use.
   *
   * If unspecified, no activation is applied.
   */
  activation?: ActivationIdentifier;
  /** Whether to apply a bias. */
  useBias?: boolean;
  /**
   * Initializer for the dense kernel weights matrix.
   */
  kernelInitializer?: InitializerIdentifier|Initializer;
  /**
   * Initializer for the bias vector.
   */
  biasInitializer?: InitializerIdentifier|Initializer;
  /**
   * If specified, defines inputShape as `[inputDim]`.
   */
  inputDim?: number;

  /**
   * Constraint for the kernel weights.
   */
  kernelConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Constraint for the bias vector.
   */
  biasConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Regularizer function applied to the dense kernel weights matrix.
   */
  kernelRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Regularizer function applied to the bias vector.
   */
  biasRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Regularizer function applied to the activation.
   */
  activityRegularizer?: RegularizerIdentifier|Regularizer;
}

/**
 * Creates a dense (fully connected) layer.
 *
 * This layer implements the operation:
 *   `output = activation(dot(input, kernel) + bias)`
 *
 * `activation` is the element-wise activation function
 *   passed as the `activation` argument.
 *
 * `kernel` is a weights matrix created by the layer.
 *
 * `bias` is a bias vector created by the layer (only applicable if `useBias`
 * is `true`).
 *
 * **Input shape:**
 *
 *   nD `Tensor` with shape: `(batchSize, ..., inputDim)`.
 *
 *   The most common situation would be
 *   a 2D input with shape `(batchSize, inputDim)`.
 *
 * **Output shape:**
 *
 *   nD tensor with shape: `(batchSize, ..., units)`.
 *
 *   For instance, for a 2D input with shape `(batchSize, inputDim)`,
 *   the output would have shape `(batchSize, units)`.
 *
 * Note: if the input to the layer has a rank greater than 2, then it is
 * flattened prior to the initial dot product with the kernel.
 */
export class Dense extends Layer {
  static className = 'Dense';
  private units: number;
  // Default activation: Linear (none).
  private activation: ActivationFn = null;
  private useBias = true;
  private kernelInitializer: Initializer;
  private biasInitializer: Initializer;
  private kernel: LayerVariable = null;
  private bias: LayerVariable = null;

  readonly DEFAULT_KERNEL_INITIALIZER: InitializerIdentifier = 'glorotNormal';
  readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier = 'zeros';
  private readonly kernelConstraint?: Constraint;
  private readonly biasConstraint?: Constraint;
  private readonly kernelRegularizer?: Regularizer;
  private readonly biasRegularizer?: Regularizer;

  constructor(config: DenseLayerConfig) {
    super(config);
    if (config.batchInputShape == null && config.inputShape == null &&
        config.inputDim != null) {
      // This logic is copied from Layer's constructor, since we can't
      // do exactly what the Python constructor does for Dense().
      let batchSize: number = null;
      if (config.batchSize != null) {
        batchSize = config.batchSize;
      }
      this.batchInputShape = [batchSize, config.inputDim];
    }

    this.units = config.units;
    this.activation = getActivation(config.activation);
    if (config.useBias != null) {
      this.useBias = config.useBias;
    }
    this.kernelInitializer = getInitializer(
        config.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.biasInitializer =
        getInitializer(config.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
    this.kernelConstraint = getConstraint(config.kernelConstraint);
    this.biasConstraint = getConstraint(config.biasConstraint);
    this.kernelRegularizer = getRegularizer(config.kernelRegularizer);
    this.biasRegularizer = getRegularizer(config.biasRegularizer);
    this.activityRegularizer = getRegularizer(config.activityRegularizer);

    this.inputSpec = [{minNDim: 2}];
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    const inputLastDim = inputShape[inputShape.length - 1];
    if (this.kernel == null) {
      this.kernel = this.addWeight(
          'kernel', [inputLastDim, this.units], null, this.kernelInitializer,
          this.kernelRegularizer, true, this.kernelConstraint);
      if (this.useBias) {
        this.bias = this.addWeight(
            'bias', [this.units], null, this.biasInitializer,
            this.biasRegularizer, true, this.biasConstraint);
      }
    }

    this.inputSpec = [{minNDim: 2, axes: {[-1]: inputLastDim}}];
    this.built = true;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    const outputShape = inputShape.slice();
    outputShape[outputShape.length - 1] = this.units;
    return outputShape;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    this.invokeCallHook(inputs, kwargs);
    // Dense layer accepts only a single input.
    const input = generic_utils.getExactlyOneTensor(inputs);
    let output = K.dot(input, this.kernel.read());
    if (this.bias != null) {
      output = K.biasAdd(output, this.bias.read());
    }
    if (this.activation != null) {
      output = this.activation(output);
    }
    return output;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      useBias: this.useBias,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint),
      biasConstraint: serializeConstraint(this.biasConstraint)
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.SerializationMap.register(Dense);

/**
 * Flattens the input. Does not affect the batch size.
 *
 * A `Flatten` layer flattens each batch in its inputs to 1D (making the output
 * 2D).
 *
 * For example:
 *
 * ```js
 * const input = tf.input({shape: [4, 3]});
 * const flattenLayer = tf.layers.flatten();
 * // Inspect the inferred output shape of the flatten layer, which
 * // equals `[null, 12]`. The 2nd dimension is 4 * 3, i.e., the result of the
 * // flattening. (The 1st dimension is the undermined batch size.)
 * console.log(JSON.stringify(flattenLayer.apply(input).shape));
 * ```
 */
export class Flatten extends Layer {
  static className = 'Flatten';
  constructor(config?: LayerConfig) {
    super(config || {});
    this.inputSpec = [{minNDim: 3}];
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    for (const dim of inputShape.slice(1)) {
      if (dim == null) {
        throw new ValueError(
            `The shape of the input to "Flatten" is not fully defined ` +
            `(got ${inputShape.slice(1)}). Make sure to pass a complete ` +
            `"input_shape" or "batch_input_shape" argument to the first ` +
            `layer in your model.`);
      }
    }
    return [inputShape[0], math_utils.arrayProd(inputShape, 1)];
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    this.invokeCallHook(inputs, kwargs);
    return K.batchFlatten(generic_utils.getExactlyOneTensor(inputs));
  }
}
serialization.SerializationMap.register(Flatten);

export interface ActivationLayerConfig extends LayerConfig {
  /**
   * Name of the activation function to use.
   */
  activation: ActivationIdentifier;
}

/**
 * Applies an activation function to an output.
 *
 * This layer applies element-wise activation function.  Other layers, notably
 * `dense` can also apply activation functions.  Use this isolated activation
 * function to extract the values before and after the
 * activation. For instance:
 *
 * ```js
 * const input = tf.input({shape: [5]});
 * const denseLayer = tf.layers.dense({units: 1});
 * const activationLayer = tf.layers.activation({activation: 'relu6'});
 *
 * // Obtain the output symbolic tensors by applying the layers in order.
 * const denseOutput = denseLayer.apply(input);
 * const activationOutput = activationLayer.apply(denseOutput);
 *
 * // Create the model based on the inputs.
 * const model = tf.model({
 *     inputs: input,
 *     outputs: [denseOutput, activationOutput]
 * });
 *
 * // Collect both outputs and print separately.
 * const [denseOut, activationOut] = model.predict(tf.randomNormal([6, 5]));
 * denseOut.print();
 * activationOut.print();
 * ```
 *
 */
export class Activation extends Layer {
  static className = 'Activation';
  activation: ActivationFn;

  constructor(config: ActivationLayerConfig) {
    super(config);
    this.supportsMasking = true;
    this.activation = getActivation(config.activation);
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    this.invokeCallHook(inputs, kwargs);
    const input = generic_utils.getExactlyOneTensor(inputs);
    return this.activation(input);
  }
}
serialization.SerializationMap.register(Activation);

export interface ReshapeLayerConfig extends LayerConfig {
  /** The target shape. Does not include the batch axis. */
  targetShape: Shape;
}

export interface RepeatVectorLayerConfig extends LayerConfig {
  /**
   * The integer number of times to repeat the input.
   */
  n: number;
}

/**
 * Repeats the input n times in a new dimension.
 *
 * ```js
 *  const model = tf.sequential();
 *  model.add(tf.layers.repeatVector({n: 4, inputShape: [2]}));
 *  const x = tf.tensor2d([[10, 20]]);
 *  // Use the model to do inference on a data point the model hasn't see
 *  model.predict(x).print();
 *  // output shape is now [batch, 2, 4]
 * ```
 */
export class RepeatVector extends Layer {
  static className = 'RepeatVector';
  readonly n: number;

  constructor(config: RepeatVectorLayerConfig) {
    super(config);
    this.n = config.n;
    this.inputSpec = [{ndim: 2}];
  }

  computeOutputShape(inputShape: Shape): Shape {
    return [inputShape[0], this.n, inputShape[1]];
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    inputs = getExactlyOneTensor(inputs);
    return K.repeat(inputs, this.n);
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      n: this.n,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.SerializationMap.register(RepeatVector);


/**
 * Reshapes an input to a certain shape.
 * TODO(cais): Code example.
 *
 * Input shape:
 *   Arbitrary: although all dimensions in the input shape must be fixed.
 *     Use the ReshapeLayerConfig field `input_shape` when using this layer
 *     as the first layer in a model.
 *
 * Output shape:
 *   [batchSize, targetShape[0], targetShape[1], ...,
 *    targetShape[targetShape.length - 1]].
 */
export class Reshape extends Layer {
  static className = 'Reshape';
  private targetShape: Shape;

  constructor(config: ReshapeLayerConfig) {
    super(config);
    this.targetShape = config.targetShape;

    // Make sure that all unknown dimensions are represented as `null`.
    for (let i = 0; i < this.targetShape.length; ++i) {
      if (this.isUnknown(this.targetShape[i])) {
        this.targetShape[i] = null;
      }
    }
  }

  private isUnknown(dim: number): boolean {
    return dim < 0 || dim == null;
  }

  /**
   * Finds and replaces a missing dimension in output shape.
   *
   * This is a near direct port of the internal Numpy function
   * `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`.
   *
   * @param inputShape: Original shape of array begin reshape.
   * @param outputShape: Target shape of the array, with at most a single
   * `null` or negative number, which indicates an underdetermined dimension
   * that should be derived from `inputShape` and the known dimensions of
   *   `outputShape`.
   * @returns: The output shape with `null` replaced with its computed value.
   * @throws: ValueError: If `inputShape` and `outputShape` do not match.
   */
  private fixUnknownDimension(inputShape: Shape, outputShape: Shape): Shape {
    const errorMsg = 'Total size of new array must be unchanged.';
    const finalShape = outputShape.slice();
    let known = 1;
    let unknown = null;
    for (let i = 0; i < finalShape.length; ++i) {
      const dim = finalShape[i];
      if (this.isUnknown(dim)) {
        if (unknown === null) {
          unknown = i;
        } else {
          throw new ValueError('Can only specifiy one unknown dimension.');
        }
      } else {
        known *= dim;
      }
    }

    const originalSize = math_utils.arrayProd(inputShape);
    if (unknown !== null) {
      if (known === 0 || originalSize % known !== 0) {
        throw new ValueError(errorMsg);
      }
      finalShape[unknown] = originalSize / known;
    } else if (originalSize !== known) {
      throw new ValueError(errorMsg);
    }

    return finalShape;
  }

  computeOutputShape(inputShape: Shape): Shape {
    let anyUnknownDims = false;
    for (let i = 0; i < inputShape.length; ++i) {
      if (this.isUnknown(inputShape[i])) {
        anyUnknownDims = true;
        break;
      }
    }

    if (anyUnknownDims) {
      return inputShape.slice(0, 1).concat(this.targetShape);
    } else {
      return inputShape.slice(0, 1).concat(
          this.fixUnknownDimension(inputShape.slice(1), this.targetShape));
    }
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    this.invokeCallHook(inputs, kwargs);
    const input = generic_utils.getExactlyOneTensor(inputs);
    const inputShape = K.shape(input);
    const outputShape = inputShape.slice(0, 1).concat(
        this.fixUnknownDimension(inputShape.slice(1), this.targetShape));
    return K.reshape(input, outputShape);
  }
}
serialization.SerializationMap.register(Reshape);
