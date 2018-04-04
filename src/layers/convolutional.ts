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

import {conv2dTranspose, Tensor, Tensor4D, tidy} from '@tensorflow/tfjs-core';
import * as _ from 'underscore';

// tslint:disable:max-line-length
import {ActivationFn, getActivation, serializeActivation} from '../activations';
import * as K from '../backend/tfjs_backend';
import {checkDataFormat, checkPaddingMode, DataFormat, PaddingMode} from '../common';
import {Constraint, ConstraintIdentifier, getConstraint, serializeConstraint} from '../constraints';
import {InputSpec, Layer, LayerConfig} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, serializeInitializer} from '../initializers';
import {getRegularizer, Regularizer, RegularizerIdentifier, serializeRegularizer} from '../regularizers';
import {Shape} from '../types';
import {ConfigDict, DType, LayerVariable} from '../types';
import {convOutputLength, deconvLength, normalizeArray} from '../utils/conv_utils';
import * as generic_utils from '../utils/generic_utils';
// tslint:enable:max-line-length

/**
 * LayerConfig for convolutional layers.
 * Applies to convolution of all ranks (e.g, Conv1D, Conv2D).
 */
export interface ConvLayerConfig extends LayerConfig {
  /**
   * The dimensions of the convolution window. If kernelSize is a number, the
   * convolutional window will be square.
   */
  kernelSize: number|number[];

  /**
   * The dimensionality of the output space (i.e. the number of filters in the
   * convolution).
   */
  filters?: number;

  /**
   * The strides of the convolution in each dimension. If strides is a number,
   * strides in both dimensions are equal.
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
   * Format of the data, which determines the ordering of the dimensions in
   * the inputs.
   *
   * `channels_last` corresponds to inputs with shape
   *   `(batch, ..., channels)`
   *
   *  `channels_first` corresponds to inputs with shape `(batch, channels,
   * ...)`.
   *
   * Defaults to `channels_last`.
   */
  dataFormat?: DataFormat;

  /**
   * The dilation rate to use for the dilated convolution in each dimension.
   * Should be an integer or array of integers.
   *
   * Currently, specifying any `dilationRate` value != 1 is incompatible with
   * specifying any `strides` value != 1.
   */
  dilationRate?: number|number[];

  /**
   * Activation function of the layer.
   *
   * If you don't specify the activation, none is applied.
   */
  activation?: string;

  /**
   * Whether the layer uses a bias vector. Defaults to false.
   */
  useBias?: boolean;

  /**
   * Initializer for the convolutional kernel weights matrix.
   */
  kernelInitializer?: InitializerIdentifier|Initializer;

  /**
   * Initializer for the bias vector.
   */
  biasInitializer?: InitializerIdentifier|Initializer;

  /**
   * Constraint for the convolutional kernel weights.
   */
  kernelConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Constraint for the bias vector.
   */
  biasConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Regularizer function applied to the kernel weights matrix.
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

  readonly DEFAULT_KERNEL_INITIALIZER: InitializerIdentifier = 'glorotNormal';
  readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier = 'zeros';

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
    this.padding = config.padding == null ? 'valid' : config.padding;
    checkPaddingMode(this.padding);
    this.dataFormat =
        config.dataFormat == null ? 'channelsLast' : config.dataFormat;
    checkDataFormat(this.dataFormat);

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
    const channelAxis =
        this.dataFormat === 'channelsFirst' ? 1 : inputShape.length - 1;
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
    const space = (this.dataFormat === 'channelsLast') ?
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
    if (this.dataFormat === 'channelsLast') {
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
 * If `activation` is not `null`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model,
 * provide the keyword argument `inputShape`
 * (Array of integers, does not include the sample axis),
 * e.g. `inputShape=[128, 128, 3]` for 128x128 RGB pictures
 * in `dataFormat='channelsLast'`.
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
 * Transposed convolutional layer (sometimes called Deconvolution).
 *
 * The need for transposed convolutions generally arises
 * from the desire to use a transformation going in the opposite direction of a
 * normal convolution, i.e., from something that has the shape of the output of
 * some convolution to something that has the shape of its input while
 * maintaining a connectivity pattern that is compatible with said convolution.
 *
 * When using this layer as the first layer in a model, provide the
 * configuration `inputShape` (`Array` of integers, does not include the sample
 * axis), e.g., `inputShape: [128, 128, 3]` for 128x128 RGB pictures in
 * `dataFormat: 'channelsLast'`.
 *
 * Input shape:
 *   4D tensor with shape:
 *   `[batch, channels, rows, cols]` if `dataFormat` is `'channelsFirst'`.
 *   or 4D tensor with shape
 *   `[batch, rows, cols, channels]` if `dataFormat` is `'channelsLast`.
 *
 * Output shape:
 *   4D tensor with shape:
 *   `[batch, filters, newRows, newCols]` if `dataFormat` is `'channelsFirst'`.
 *   or 4D tensor with shape:
 *   `[batch, newRows, newCols, filters]` if `dataFormat` is `'channelsLast'`.
 *
 * References:
 *   - [A guide to convolution arithmetic for deep
 * learning](https://arxiv.org/abs/1603.07285v1)
 *   - [Deconvolutional
 * Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
 */
export class Conv2DTranspose extends Conv2D {
  inputSpec: InputSpec[];

  constructor(config: ConvLayerConfig) {
    super(config);
    this.inputSpec = [new InputSpec({ndim: 4})];

    if (this.padding !== 'same' && this.padding !== 'valid') {
      throw new ValueError(
          `Conv2DTranspose currently supports only padding modes 'same' ` +
          `and 'valid', but received padding mode ${this.padding}`);
    }
  }

  build(inputShape: Shape|Shape[]): void {
    inputShape = generic_utils.getExactlyOneShape(inputShape);

    if (inputShape.length !== 4) {
      throw new ValueError(
          'Input should have rank 4; Received input shape: ' +
          JSON.stringify(inputShape));
    }

    const channelAxis =
        this.dataFormat === 'channelsFirst' ? 1 : inputShape.length - 1;
    if (inputShape[channelAxis] == null) {
      throw new ValueError(
          'The channel dimension of the inputs should be defined. ' +
          'Found `None`.');
    }
    const inputDim = inputShape[channelAxis];
    const kernelShape = this.kernelSize.concat([this.filters, inputDim]);

    this.kernel = this.addWeight(
        'kernel', kernelShape, DType.float32, this.kernelInitializer,
        this.kernelRegularizer, true, this.kernelConstraint);
    if (this.useBias) {
      this.bias = this.addWeight(
          'kernel', [this.filters], DType.float32, this.biasInitializer,
          this.biasRegularizer, true, this.biasConstraint);
    }

    // Set input spec.
    this.inputSpec =
        [new InputSpec({ndim: 4, axes: {[channelAxis]: inputDim}})];
    this.built = true;
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    return tidy(() => {
      let input = generic_utils.getExactlyOneTensor(inputs);
      if (input.shape.length !== 4) {
        throw new ValueError(
            `Conv2DTranspose.call() expects input tensor to be rank-4, but ` +
            `received a tensor of rank-${input.shape.length}`);
      }

      const inputShape = input.shape;
      const batchSize = inputShape[0];

      let hAxis: number;
      let wAxis: number;
      if (this.dataFormat === 'channelsFirst') {
        hAxis = 2;
        wAxis = 3;
      } else {
        hAxis = 1;
        wAxis = 2;
      }

      const height = inputShape[hAxis];
      const width = inputShape[wAxis];
      const kernelH = this.kernelSize[0];
      const kernelW = this.kernelSize[1];
      const strideH = this.strides[0];
      const strideW = this.strides[1];

      // Infer the dynamic output shape.
      const outHeight = deconvLength(height, strideH, kernelH, this.padding);
      const outWidth = deconvLength(width, strideW, kernelW, this.padding);

      // Porting Note: We don't branch based on `this.dataFormat` here, because
      //   the tjfs-core function `conv2dTranspose` called below always assumes
      //   channelsLast.
      const outputShape: [number, number, number, number] =
          [batchSize, outHeight, outWidth, this.filters];

      if (this.dataFormat !== 'channelsLast') {
        input = K.transpose(input, [0, 2, 3, 1]);
      }
      let outputs = conv2dTranspose(
          input as Tensor4D, this.kernel.read() as Tensor4D, outputShape,
          this.strides as [number, number], this.padding as 'same' | 'valid');
      if (this.dataFormat !== 'channelsLast') {
        outputs = K.transpose(outputs, [0, 3, 1, 2]) as Tensor4D;
      }

      if (this.bias != null) {
        outputs =
            K.biasAdd(outputs, this.bias.read(), this.dataFormat) as Tensor4D;
      }
      if (this.activation != null) {
        outputs = this.activation(outputs) as Tensor4D;
      }
      return outputs;
    });
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    const outputShape = inputShape.slice();

    let channelAxis: number;
    let heightAxis: number;
    let widthAxis: number;
    if (this.dataFormat === 'channelsFirst') {
      channelAxis = 1;
      heightAxis = 2;
      widthAxis = 3;
    } else {
      channelAxis = 3;
      heightAxis = 1;
      widthAxis = 2;
    }

    const kernelH = this.kernelSize[0];
    const kernelW = this.kernelSize[1];
    const strideH = this.strides[0];
    const strideW = this.strides[1];

    outputShape[channelAxis] = this.filters;
    outputShape[heightAxis] =
        deconvLength(outputShape[heightAxis], strideH, kernelH, this.padding);
    outputShape[widthAxis] =
        deconvLength(outputShape[widthAxis], strideW, kernelW, this.padding);
    return outputShape;
  }

  getConfig(): ConfigDict {
    const config = super.getConfig();
    delete config['dilationRate'];
    return config;
  }
}
generic_utils.ClassNameMap.register('Conv2DTranspose', Conv2DTranspose);

/**
 * 1D convolution layer (e.g., temporal convolution).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input over a single spatial (or temporal) dimension
 * to produce a tensor of outputs.
 *
 * If `use_bias` is True, a bias vector is created and added to the outputs.
 *
 * If `activation` is not `null`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model, provide an `inputShape`
 * argument `Array` or `null`.
 *
 * For example, `inputShape` would be:
 * - `[10, 128]` for sequences of 10 vectors of 128-dimensional vectors
 * - `[null, 128]` for variable-length sequences of 128-dimensional vectors.
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
