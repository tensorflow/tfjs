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

import * as activations from '../activations';
import * as K from '../backend/deeplearnjs_backend';
import {DataFormat, PaddingMode} from '../common';
import * as constraints from '../constraints';
import {Layer, LayerConfig} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import * as initializers from '../initializers';
import * as regularizers from '../regularizers';
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
   * kernelSize: An integer or an array of integers, specifying the
   *   dimensions of the convolution window.
   */
  kernelSize: number|number[];

  /**
   * filters: Integer, the dimensionality of the output space
   *   (i.e. the number output of filters in the convolution).
   */
  filters?: number;

  /**
   * strides: An integer or tuple/list of n integers,
   *   specifying the strides of the convolution.
   *   Specifying any stride value != 1 is incompatible with specifying
   *   any `dilationRate` value != 1.
   */
  strides?: number|number[];

  /**
   * padding: Padding mode, e.g., VALID.
   */
  padding?: PaddingMode;

  /**
   * dataFormat: Format of the data, e.g., CHANNEL_LAST.
   *   The ordering of the dimensions in the inputs.
   *   `channels_last` corresponds to inputs with shape
   *   `(batch, ..., channels)` while `channels_first` corresponds to
   *   inputs with shape `(batch, channels, ...)`.
   *   Defaults to "channels_last".
   */
  dataFormat?: DataFormat;

  /**
   * dilation_rate: An integer or array of integers, specifying
   *   the dilation rate to use for dilated convolution.
   *   Currently, specifying any `dilationRate` value != 1 is
   *   incompatible with specifying any `strides` value != 1.
   */
  dilationRate?: number|number[];

  /**
   * activation: Activation function to use
   *   If you don't specify anything, no activation is applied
   *   (ie. "linear" activation: `a(x) = x`).
   */
  activation?: string;

  /**
   * useBias: Boolean, whether the layer uses a bias vector.
   */
  useBias?: boolean;

  /**
   * kernelInitializer: Initializer for the `kernel` weights matrix
   */
  kernelInitializer?: string|initializers.Initializer;

  /**
   * biasInitializer: Initializer for the bias vector
   */
  biasInitializer?: string|initializers.Initializer;

  /**
   * kernelConstraint: Constraint for the kernel weights
   */
  kernelConstraint?: string|constraints.Constraint;

  /**
   * biasConstraint: Constraint for the bias vector
   */
  biasConstraint?: string|constraints.Constraint;

  /**
   * kernelRegularizer:  Regularizer function applied to the `kernel` weights
   * matrix
   */
  kernelRegularizer?: string|regularizers.Regularizer;

  /**
   * biasRegularizer:  Regularizer function applied to the bias vector
   */
  biasRegularizer?: string|regularizers.Regularizer;

  /**
   * activityRegularizer:  Regularizer function applied to the activation
   */
  activityRegularizer?: string|regularizers.Regularizer;
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
  protected readonly activation: activations.ActivationFn;
  protected readonly useBias: boolean;
  protected readonly kernelInitializer?: initializers.Initializer;
  protected readonly biasInitializer?: initializers.Initializer;
  protected readonly kernelConstraint?: constraints.Constraint;
  protected readonly biasConstraint?: constraints.Constraint;
  protected readonly kernelRegularizer?: regularizers.Regularizer;
  protected readonly biasRegularizer?: regularizers.Regularizer;

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
    this.activation = activations.get(config.activation);
    this.useBias = config.useBias == null ? true : config.useBias;
    this.kernelInitializer = initializers.get(
        config.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.biasInitializer = initializers.get(
        config.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
    this.kernelConstraint = constraints.get(config.kernelConstraint);
    this.biasConstraint = constraints.get(config.biasConstraint);
    this.kernelRegularizer = regularizers.get(config.kernelRegularizer);
    this.biasRegularizer = regularizers.get(config.biasRegularizer);
    this.activityRegularizer = regularizers.get(config.activityRegularizer);
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
      activation: activations.serialize(this.activation),
      useBias: this.useBias,
      kernelInitializer: initializers.serialize(this.kernelInitializer),
      biasInitializer: initializers.serialize(this.biasInitializer),
      kernelRegularizer: regularizers.serialize(this.kernelRegularizer),
      biasRegularizer: regularizers.serialize(this.biasRegularizer),
      activityRegularizer: regularizers.serialize(this.activityRegularizer),
      kernelConstraint: constraints.serialize(this.kernelConstraint),
      biasConstraint: constraints.serialize(this.biasConstraint)
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
 * with the layer input to produce a tensor of outputs. If `useBias` is True,
 * a bias vector is created and added to the outputs. Finally, if
 * `activation` is not `None`, it is applied to the outputs as well.
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
