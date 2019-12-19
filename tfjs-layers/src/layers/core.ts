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

import {any, notEqual, serialization, Tensor, tidy, transpose, util} from '@tensorflow/tfjs-core';

import {Activation as ActivationFn, getActivation, serializeActivation} from '../activations';
import * as K from '../backend/tfjs_backend';
import {Constraint, ConstraintIdentifier, getConstraint, serializeConstraint} from '../constraints';
import {DisposeResult, InputSpec, Layer, LayerArgs} from '../engine/topology';
import {ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, serializeInitializer} from '../initializers';
import {ActivationIdentifier} from '../keras_format/activation_config';
import {DataFormat, Shape} from '../keras_format/common';
import {LayerConfig} from '../keras_format/topology_config';
import {getRegularizer, Regularizer, RegularizerIdentifier, serializeRegularizer} from '../regularizers';
import {Kwargs} from '../types';
import {assertPositiveInteger, mapActivationToFusedKernel} from '../utils/generic_utils';
import {arrayProd, range} from '../utils/math_utils';
import {getExactlyOneShape, getExactlyOneTensor} from '../utils/types_utils';
import {LayerVariable} from '../variables';

export declare interface DropoutLayerArgs extends LayerArgs {
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

export class Dropout extends Layer {
  /** @nocollapse */
  static className = 'Dropout';
  private readonly rate: number;
  private readonly noiseShape: number[];
  private readonly seed: number;

  constructor(args: DropoutLayerArgs) {
    super(args);
    this.rate = Math.max(Math.min(args.rate, 1), 0);
    // So that the scalar doesn't get tidied up between executions.
    this.noiseShape = args.noiseShape;
    this.seed = args.seed;
    this.supportsMasking = true;
  }

  protected getNoiseShape(input: Tensor): Shape {
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
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      const input = getExactlyOneTensor(inputs);
      if (0 < this.rate && this.rate < 1) {
        const training =
            kwargs['training'] == null ? false : kwargs['training'];
        const noiseShape = this.getNoiseShape(input);
        const output = K.inTrainPhase(
            () => K.dropout(input, this.rate, noiseShape, this.seed),
            () => input, training);
        return output;
      }
      return inputs;
    });
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

  dispose(): DisposeResult {
    return super.dispose();
  }
}
serialization.registerClass(Dropout);

export declare interface DenseLayerArgs extends LayerArgs {
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

export interface SpatialDropout1DLayerConfig extends LayerConfig {
  /** Float between 0 and 1. Fraction of the input units to drop. */
  rate: number;

  /** An integer to use as random seed. */
  seed?: number;
}

export class SpatialDropout1D extends Dropout {
  /** @nocollapse */
  static className = 'SpatialDropout1D';

  constructor(args: SpatialDropout1DLayerConfig) {
    super(args);
    this.inputSpec = [{ndim: 3}];
  }

  protected getNoiseShape(input: Tensor): Shape {
    const inputShape = input.shape;
    return [inputShape[0], 1, inputShape[2]];
  }
}
serialization.registerClass(SpatialDropout1D);

export class Dense extends Layer {
  /** @nocollapse */
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

  constructor(args: DenseLayerArgs) {
    super(args);
    if (args.batchInputShape == null && args.inputShape == null &&
        args.inputDim != null) {
      // This logic is copied from Layer's constructor, since we can't
      // do exactly what the Python constructor does for Dense().
      let batchSize: number = null;
      if (args.batchSize != null) {
        batchSize = args.batchSize;
      }
      this.batchInputShape = [batchSize, args.inputDim];
    }

    this.units = args.units;
    assertPositiveInteger(this.units, 'units');
    this.activation = getActivation(args.activation);
    if (args.useBias != null) {
      this.useBias = args.useBias;
    }
    this.kernelInitializer = getInitializer(
        args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.biasInitializer =
        getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
    this.kernelConstraint = getConstraint(args.kernelConstraint);
    this.biasConstraint = getConstraint(args.biasConstraint);
    this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
    this.biasRegularizer = getRegularizer(args.biasRegularizer);
    this.activityRegularizer = getRegularizer(args.activityRegularizer);
    this.supportsMasking = true;

    this.inputSpec = [{minNDim: 2}];
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = getExactlyOneShape(inputShape);
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
    inputShape = getExactlyOneShape(inputShape);
    const outputShape = inputShape.slice();
    outputShape[outputShape.length - 1] = this.units;
    return outputShape;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      // Dense layer accepts only a single input.
      const input = getExactlyOneTensor(inputs);
      const fusedActivationName =
          mapActivationToFusedKernel(this.activation.getClassName());
      let output: Tensor;

      if (fusedActivationName != null) {
        output = K.dot(
            input, this.kernel.read(), fusedActivationName,
            this.bias ? this.bias.read() : null);
      } else {
        output = K.dot(input, this.kernel.read());
        if (this.bias != null) {
          output = K.biasAdd(output, this.bias.read());
        }
        if (this.activation != null) {
          output = this.activation.apply(output);
        }
      }

      return output;
    });
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
serialization.registerClass(Dense);

export declare interface FlattenLayerArgs extends LayerArgs {
  /** Image data format: channeLast (default) or channelFirst. */
  dataFormat?: DataFormat;
}

export class Flatten extends Layer {
  private dataFormat: DataFormat;

  /** @nocollapse */
  static className = 'Flatten';
  constructor(args?: FlattenLayerArgs) {
    args = args || {};
    super(args);
    this.inputSpec = [{minNDim: 3}];
    this.dataFormat = args.dataFormat;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    for (const dim of inputShape.slice(1)) {
      if (dim == null) {
        throw new ValueError(
            `The shape of the input to "Flatten" is not fully defined ` +
            `(got ${inputShape.slice(1)}). Make sure to pass a complete ` +
            `"input_shape" or "batch_input_shape" argument to the first ` +
            `layer in your model.`);
      }
    }
    return [inputShape[0], arrayProd(inputShape, 1)];
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);

      let input = getExactlyOneTensor(inputs);
      if (this.dataFormat === 'channelsFirst' && input.rank > 1) {
        const permutation: number[] = [0];
        for (let i = 2; i < input.rank; ++i) {
          permutation.push(i);
        }
        permutation.push(1);
        input = input.transpose(permutation);
      }

      return K.batchFlatten(input);
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {};
    if (this.dataFormat != null) {
      config['dataFormat'] = this.dataFormat;
    }
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(Flatten);

export declare interface ActivationLayerArgs extends LayerArgs {
  /**
   * Name of the activation function to use.
   */
  activation: ActivationIdentifier;
}

export class Activation extends Layer {
  /** @nocollapse */
  static className = 'Activation';
  activation: ActivationFn;

  constructor(args: ActivationLayerArgs) {
    super(args);
    this.supportsMasking = true;
    this.activation = getActivation(args.activation);
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      const input = getExactlyOneTensor(inputs);
      return this.activation.apply(input);
    });
  }

  getConfig(): serialization.ConfigDict {
    const config = {activation: serializeActivation(this.activation)};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(Activation);

export declare interface ReshapeLayerArgs extends LayerArgs {
  /** The target shape. Does not include the batch axis. */
  targetShape: Shape;
}

export declare interface RepeatVectorLayerArgs extends LayerArgs {
  /**
   * The integer number of times to repeat the input.
   */
  n: number;
}

export class RepeatVector extends Layer {
  /** @nocollapse */
  static className = 'RepeatVector';
  readonly n: number;

  constructor(args: RepeatVectorLayerArgs) {
    super(args);
    this.n = args.n;
    this.inputSpec = [{ndim: 2}];
  }

  computeOutputShape(inputShape: Shape): Shape {
    return [inputShape[0], this.n, inputShape[1]];
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      inputs = getExactlyOneTensor(inputs);
      return K.repeat(inputs, this.n);
    });
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
serialization.registerClass(RepeatVector);

export class Reshape extends Layer {
  /** @nocollapse */
  static className = 'Reshape';
  private targetShape: Shape;

  constructor(args: ReshapeLayerArgs) {
    super(args);
    this.targetShape = args.targetShape;

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

    const originalSize = arrayProd(inputShape);
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
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      const input = getExactlyOneTensor(inputs);
      const inputShape = input.shape;
      const outputShape = inputShape.slice(0, 1).concat(
          this.fixUnknownDimension(inputShape.slice(1), this.targetShape));
      return input.reshape(outputShape);
    });
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      targetShape: this.targetShape,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(Reshape);

export declare interface PermuteLayerArgs extends LayerArgs {
  /**
   * Array of integers. Permutation pattern. Does not include the
   * sample (batch) dimension. Index starts at 1.
   * For instance, `[2, 1]` permutes the first and second dimensions
   * of the input.
   */
  dims: number[];
}

export class Permute extends Layer {
  /** @nocollapse */
  static className = 'Permute';
  readonly dims: number[];
  private readonly dimsIncludingBatch: number[];

  constructor(args: PermuteLayerArgs) {
    super(args);
    if (args.dims == null) {
      throw new Error(
          'Required configuration field `dims` is missing during Permute ' +
          'constructor call.');
    }
    if (!Array.isArray(args.dims)) {
      throw new Error(
          'Permute constructor requires `dims` to be an Array, but received ' +
          `${args.dims} instead.`);
    }

    // Check the validity of the permutation indices.
    const expectedSortedIndices = range(1, args.dims.length + 1);
    if (!util.arraysEqual(args.dims.slice().sort(), expectedSortedIndices)) {
      throw new Error(
          'Invalid permutation `dims`: ' + JSON.stringify(args.dims) +
          ' `dims` must contain consecutive integers starting from 1.');
    }

    this.dims = args.dims;
    this.dimsIncludingBatch = [0].concat(this.dims);
    this.inputSpec = [new InputSpec({ndim: this.dims.length + 1})];
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    const outputShape = inputShape.slice();
    this.dims.forEach((dim: number, i: number) => {
      outputShape[i + 1] = (inputShape as Shape)[dim];
    });
    return outputShape;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return transpose(getExactlyOneTensor(inputs), this.dimsIncludingBatch);
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      dims: this.dims,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(Permute);

export declare interface MaskingArgs extends LayerArgs {
  /**
   * Masking Value. Defaults to `0.0`.
   */
  maskValue?: number;
}

export class Masking extends Layer {
  /** @nocollapse */
  static className = 'Masking';
  maskValue: number;

  constructor(args?: MaskingArgs) {
    super(args == null ? {} : args);
    this.supportsMasking = true;
    if (args != null) {
      this.maskValue = args.maskValue == null ? 0 : args.maskValue;
    } else {
      this.maskValue = 0;
    }
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig() {
    const baseConfig = super.getConfig();
    const config = {maskValue: this.maskValue};
    Object.assign(config, baseConfig);
    return config;
  }

  computeMask(inputs: Tensor|Tensor[], mask?: Tensor|Tensor[]): Tensor {
    const input = getExactlyOneTensor(inputs);
    const axis = -1;
    return any(notEqual(input, this.maskValue), axis);
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      const input = getExactlyOneTensor(inputs);
      const axis = -1;
      const keepDims = true;
      const booleanMask = any(notEqual(input, this.maskValue), axis, keepDims);
      const output = input.mul(booleanMask.asType(input.dtype));
      return output;
    });
  }
}
serialization.registerClass(Masking);
