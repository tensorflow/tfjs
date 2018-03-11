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
 * TensorFlow.js Layers: Convolutional Layers
 */

import {Tensor} from 'deeplearn';
import * as _ from 'underscore';

import {ActivationFn, getActivation, serializeActivation} from '../activations';
import * as K from '../backend/deeplearnjs_backend';
import {DataFormat, PaddingMode} from '../common';
import {Constraint, getConstraint, serializeConstraint} from '../constraints';
import {Layer, LayerConfig} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {getInitializer, Initializer, serializeInitializer} from '../initializers';
import {getRegularizer, Regularizer, serializeRegularizer} from '../regularizers';
import {Shape} from '../types';
import {ConfigDict, LayerVariable} from '../types';
import {convOutputLength, normalizeArray} from '../utils/conv_utils';
import * as generic_utils from '../utils/generic_utils';

/**
 * LayerConfig for convoluational layers.
 * Applies to convolution of all ranks (e.g, Conv1D, Conv2D).
 */
export interface ConvLayerConfig extends LayerConfig {
  /**
   * The dimensions of the convolution window. If kernelSize is a number, the
   * convolutional window will be square.
   */
  kernelSize: number|number[];

  /**
   * The dimensionality of the output space (i.e. the number output of
   * filters in the convolution).
   */
  filters?: number;

  /**
   * The strides of the convolution. If strides is a number, strides in both
   * dimensions are equal.
   *
   * Specifying any stride value != 1 is incompatible with specifying any
   * `dilationRate` value != 1.
   */
  strides?: number|number[];

  /**
   * Padding mode.
   */
  padding?: PaddingMode;

  /**
   * Format of the data, e.g., CHANNEL_LAST.
   *   The ordering of the dimensions in the inputs.
   *   `channels_last` corresponds to inputs with shape
   *   `(batch, ..., channels)` while `channels_first` corresponds to
   *   inputs with shape `(batch, channels, ...)`.
   *   Defaults to "channels_last".
   */
  dataFormat?: DataFormat;

  /**
   * An integer or array of integers, specifying
   *   the dilation rate to use for dilated convolution.
   *   Currently, specifying any `dilationRate` value != 1 is
   *   incompatible with specifying any `strides` value != 1.
   */
  dilationRate?: number|number[];

  /**
   * Activation function of the layer.
   *
   * If you don't specify the activation, none is applied
   *   (ie. "linear" activation: `a(x) = x`).
   */
  activation?: string;

  /**
   * Whether the layer uses a bias vector. Defaults to false.
   */
  useBias?: boolean;

  /**
   * Initializer for the `kernel` weights matrix.
   */
  kernelInitializer?: string|Initializer;

  /**
   * Initializer for the bias vector.
   */
  biasInitializer?: string|Initializer;

  /**
   * Constraint for the kernel weights.
   */
  kernelConstraint?: string|Constraint;

  /**
   * Constraint for the bias vector.
   */
  biasConstraint?: string|Constraint;

  /**
   * Regularizer function applied to the `kernel` weights matrix.
   */
  kernelRegularizer?: string|Regularizer;

  /**
   * Regularizer function applied to the bias vector.
   */
  biasRegularizer?: string|Regularizer;

  /**
   * Regularizer function applied to the activation.
   */
  activityRegularizer?: string|Regularizer;
}

/**
 * Abstract nD convolution layer.
 */
export abstract class Conv extends Layer {
  protected readonly rank: number;
  protected readonly filters: number;
  protected readonly kernelSize: number[];
  protected readonly strides: number[];
  protected readonly padding: PaddingMode;
  protected readonly dataFormat: DataFormat;
  protected readonly dilationRate: number|number[];
  protected readonly activation: ActivationFn;
  protected readonly useBias: boolean;
  protected readonly kernelInitializer?: Initializer;
  protected readonly biasInitializer?: Initializer;
  protected readonly kernelConstraint?: Constraint;
  protected readonly biasConstraint?: Constraint;
  protected readonly kernelRegularizer?: Regularizer;
  protected readonly biasRegularizer?: Regularizer;

  protected kernel: LayerVariable = null;
  protected bias: LayerVariable = null;

  readonly DEFAULT_KERNEL_INITIALIZER = 'GlorotNormal';
  readonly DEFAULT_BIAS_INITIALIZER = 'Zeros';

  constructor(rank: number, config: ConvLayerConfig) {
    super(config);
    this.rank = rank;
    if (this.rank !== 1 && this.rank !== 2) {
      throw new NotImplementedError(
          `Convolution layer for rank other than 1 or 2 (${this.rank}) is ` +
          `not implemented yet.`);
    }

    this.filters = config.filters;
    this.kernelSize = normalizeArray(config.kernelSize, rank, 'kernelSize');
    this.strides = normalizeArray(
        config.strides == null ? 1 : config.strides, rank, 'strides');
    this.padding = config.padding == null ? PaddingMode.VALID : config.padding;
    this.dataFormat =
        config.dataFormat == null ? DataFormat.CHANNEL_LAST : config.dataFormat;

    this.dilationRate = config.dilationRate == null ? 1 : config.dilationRate;
    if (!(this.dilationRate === 1 ||
          (Array.isArray(this.dilationRate) &&
           _.isEqual(_.uniq(this.dilationRate), [1])))) {
      throw new NotImplementedError(
          'Non-default dilation is not implemented for convolution layers ' +
          'yet.');
    }
    this.activation = getActivation(config.activation);
    this.useBias = config.useBias == null ? true : config.useBias;
    this.kernelInitializer = getInitializer(
        config.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.biasInitializer =
        getInitializer(config.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
    this.kernelConstraint = getConstraint(config.kernelConstraint);
    this.biasConstraint = getConstraint(config.biasConstraint);
    this.kernelRegularizer = getRegularizer(config.kernelRegularizer);
    this.biasRegularizer = getRegularizer(config.biasRegularizer);
    this.activityRegularizer = getRegularizer(config.activityRegularizer);
  }

  build(inputShape: Shape|Shape[]): void {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    const channelAxis = this.dataFormat === DataFormat.CHANNEL_FIRST ?
        1 :
        inputShape.length - 1;
    if (inputShape[channelAxis] == null) {
      throw new ValueError(
          `The channel dimension of the input should be defined. ` +
          `Found ${inputShape[channelAxis]}`);
    }
    const inputDim = inputShape[channelAxis];

    const kernelShape = this.kernelSize.concat([inputDim, this.filters]);
    this.kernel = this.addWeight(
        'kernel', kernelShape, null, this.kernelInitializer,
        this.kernelRegularizer, true, this.kernelConstraint);
    if (this.useBias) {
      this.bias = this.addWeight(
          'bias', [this.filters], null, this.biasInitializer,
          this.biasRegularizer, true, this.biasConstraint);
    }

    this.inputSpec = [{ndim: this.rank + 2, axes: {[channelAxis]: inputDim}}];
    this.built = true;
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    inputs = generic_utils.getExactlyOneTensor(inputs);
    let outputs: Tensor;
    const biasValue = this.bias == null ? null : this.bias.read();
    if (this.rank === 1) {
      outputs = K.conv1dWithBias(
          inputs, this.kernel.read(), biasValue, this.strides[0], this.padding,
          this.dataFormat);
    } else if (this.rank === 2) {
      outputs = K.conv2dWithBias(
          inputs, this.kernel.read(), biasValue, this.strides, this.padding,
          this.dataFormat);
    } else if (this.rank === 3) {
      throw new NotImplementedError('3D convolution is not implemented yet.');
    }

    if (this.activation != null) {
      outputs = this.activation(outputs);
    }
    return outputs;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    const newSpace: number[] = [];
    const space = (this.dataFormat === DataFormat.CHANNEL_LAST) ?
        inputShape.slice(1, inputShape.length - 1) :
        inputShape.slice(2);
    for (let i = 0; i < space.length; ++i) {
      const newDim = convOutputLength(
          space[i], this.kernelSize[i], this.padding, this.strides[i],
          typeof this.dilationRate === 'number' ? this.dilationRate :
                                                  this.dilationRate[i]);
      newSpace.push(newDim);
    }

    let outputShape = [inputShape[0]];
    if (this.dataFormat === DataFormat.CHANNEL_LAST) {
      outputShape = outputShape.concat(newSpace);
      outputShape.push(this.filters);
    } else {
      outputShape.push(this.filters);
      outputShape = outputShape.concat(newSpace);
    }
    return outputShape;
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      rank: this.rank,
      filters: this.filters,
      kernelSize: this.kernelSize,
      strides: this.strides,
      padding: this.padding,
      dataFormat: this.dataFormat,
      dilationRate: this.dilationRate,
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


/**
 * 2D convolution layer (e.g. spatial convolution over images).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input to produce a tensor of outputs.
 *
 * If `useBias` is True, a bias vector is created and added to the outputs.
 *
 * If `activation` is not `None`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model,
 * provide the keyword argument `inputShape`
 * (Array of integers, does not include the sample axis),
 * e.g. `inputShape=[128, 128, 3]` for 128x128 RGB pictures
 * in `dataFormat=DataFormat.CHANNEL_LAST`.
 */
export class Conv2D extends Conv {
  constructor(config: ConvLayerConfig) {
    super(2, config);
  }

  getConfig(): ConfigDict {
    const config = super.getConfig();
    delete config['rank'];
    return config;
  }
}
generic_utils.ClassNameMap.register('Conv2D', Conv2D);

/**
 * 1D convolution layer (e.g., temporal convolution).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input over a single spatial (or temporal) dimension
 * to produce a tensor of outputs.
 * If `use_bias` is True, a bias vector is created and added to the outputs.
 * Finally, if `activation` is not `null` or `undefined`,
 * it is applied to the outputs as well.
 * When using this layer as the first layer in a model,
 * provide an `inputShape` argument Array or `null`, e.g.
 * `[10, 128]` for sequences of 10 vectors of 128-dimensional vectors,
 * or `[null, 128]` for variable-length sequences of 128-dimensional vectors.
 */
export class Conv1D extends Conv {
  constructor(config: ConvLayerConfig) {
    super(1, config);
    this.inputSpec = [{ndim: 3}];
  }

  getConfig(): ConfigDict {
    const config = super.getConfig();
    delete config['rank'];
    delete config['dataFormat'];
    return config;
  }
}
generic_utils.ClassNameMap.register('Conv1D', Conv1D);
