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

// tslint:disable:max-line-length
import {conv2dTranspose, separableConv2d, serialization, Tensor, Tensor4D, tidy, transpose} from '@tensorflow/tfjs-core';

import {ActivationFn, getActivation, serializeActivation} from '../activations';
import * as K from '../backend/tfjs_backend';
import {checkDataFormat, checkPaddingMode, DataFormat, PaddingMode} from '../common';
import {Constraint, ConstraintIdentifier, getConstraint, serializeConstraint} from '../constraints';
import {InputSpec, Layer, LayerConfig} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, serializeInitializer} from '../initializers';
import {getRegularizer, Regularizer, RegularizerIdentifier, serializeRegularizer} from '../regularizers';
import {Kwargs, Shape} from '../types';
import {DType} from '../types';
import {convOutputLength, deconvLength, normalizeArray} from '../utils/conv_utils';
import * as generic_utils from '../utils/generic_utils';
import {LayerVariable} from '../variables';
// tslint:enable:max-line-length

/**
 * Base LayerConfig for depthwise and non-depthwise convolutional layers.
 */
export interface BaseConvLayerConfig extends LayerConfig {
  /**
   * The dimensions of the convolution window. If kernelSize is a number, the
   * convolutional window will be square.
   */
  kernelSize: number|number[];

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
   * Should be an integer or array of two integers.
   *
   * Currently, specifying any `dilationRate` value != 1 is incompatible with
   * specifying any `strides` value != 1.
   */
  dilationRate?: number|[number]|[number, number];

  /**
   * Activation function of the layer.
   *
   * If you don't specify the activation, none is applied.
   */
  activation?: string;

  /**
   * Whether the layer uses a bias vector. Defaults to `true`.
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
 * LayerConfig for non-depthwise convolutional layers.
 * Applies to non-depthwise convolution of all ranks (e.g, Conv1D, Conv2D).
 */
export interface ConvLayerConfig extends BaseConvLayerConfig {
  /**
   * The dimensionality of the output space (i.e. the number of filters in the
   * convolution).
   */
  filters: number;
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
  protected readonly dilationRate: number|[number]|[number, number];
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
    if (this.rank === 1 &&
        (Array.isArray(this.dilationRate) &&
         (this.dilationRate as number[]).length !== 1)) {
      throw new ValueError(
          `dilationRate must be a number or an array of a single number ` +
          `for 1D convolution, but received ` +
          `${JSON.stringify(this.dilationRate)}`);
    }
    if (this.rank === 2) {
      if (typeof this.dilationRate === 'number') {
        this.dilationRate = [this.dilationRate, this.dilationRate];
      } else if (this.dilationRate.length !== 2) {
        throw new ValueError(
            `dilationRate must be a number or array of two numbers for 2D ` +
            `convolution, but received ${JSON.stringify(this.dilationRate)}`);
      }
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

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    inputs = generic_utils.getExactlyOneTensor(inputs);
    let outputs: Tensor;
    const biasValue = this.bias == null ? null : this.bias.read();

    if (this.rank === 1) {
      outputs = K.conv1dWithBias(
          inputs, this.kernel.read(), biasValue, this.strides[0], this.padding,
          this.dataFormat, this.dilationRate as number);
    } else if (this.rank === 2) {
      // TODO(cais): Move up to constructor.
      outputs = K.conv2dWithBias(
          inputs, this.kernel.read(), biasValue, this.strides, this.padding,
          this.dataFormat, this.dilationRate as [number, number]);
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

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
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
  static className = 'Conv2D';
  constructor(config: ConvLayerConfig) {
    super(2, config);
  }

  getConfig(): serialization.ConfigDict {
    const config = super.getConfig();
    delete config['rank'];
    return config;
  }
}
serialization.SerializationMap.register(Conv2D);

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
  static className = 'Conv2DTranspose';
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
          'bias', [this.filters], DType.float32, this.biasInitializer,
          this.biasRegularizer, true, this.biasConstraint);
    }

    // Set input spec.
    this.inputSpec =
        [new InputSpec({ndim: 4, axes: {[channelAxis]: inputDim}})];
    this.built = true;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
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
        input = transpose(input, [0, 2, 3, 1]);
      }
      let outputs = conv2dTranspose(
          input as Tensor4D, this.kernel.read() as Tensor4D, outputShape,
          this.strides as [number, number], this.padding as 'same' | 'valid');
      if (this.dataFormat !== 'channelsLast') {
        outputs = transpose(outputs, [0, 3, 1, 2]) as Tensor4D;
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

  getConfig(): serialization.ConfigDict {
    const config = super.getConfig();
    delete config['dilationRate'];
    return config;
  }
}
serialization.SerializationMap.register(Conv2DTranspose);


export interface SeparableConvLayerConfig extends ConvLayerConfig {
  /**
   * The number of depthwise convolution output channels for each input
   * channel.
   * The total number of depthwise convolution output channels will be equal to
   * `filtersIn * depthMultiplier`.
   * Default: 1.
   */
  depthMultiplier?: number;

  /**
   * Initializer for the depthwise kernel matrix.
   */
  depthwiseInitializer?: InitializerIdentifier|Initializer;

  /**
   * Initializer for the pointwise kernel matrix.
   */
  pointwiseInitializer?: InitializerIdentifier|Initializer;

  /**
   * Regularizer function applied to the depthwise kernel matrix.
   */
  depthwiseRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Regularizer function applied to the pointwise kernel matrix.
   */
  pointwiseRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Constraint function applied to the depthwise kernel matrix.
   */
  depthwiseConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Constraint function applied to the pointwise kernel matrix.
   */
  pointwiseConstraint?: ConstraintIdentifier|Constraint;
}


export class SeparableConv extends Conv {
  static className = 'SeparableConv';

  readonly depthMultiplier: number;

  protected readonly depthwiseInitializer?: Initializer;
  protected readonly depthwiseRegularizer?: Regularizer;
  protected readonly depthwiseConstraint?: Constraint;
  protected readonly pointwiseInitializer?: Initializer;
  protected readonly pointwiseRegularizer?: Regularizer;
  protected readonly pointwiseConstraint?: Constraint;

  readonly DEFAULT_DEPTHWISE_INITIALIZER: InitializerIdentifier =
      'glorotUniform';
  readonly DEFAULT_POINTWISE_INITIALIZER: InitializerIdentifier =
      'glorotUniform';

  protected depthwiseKernel: LayerVariable = null;
  protected pointwiseKernel: LayerVariable = null;

  constructor(rank: number, config?: SeparableConvLayerConfig) {
    super(rank, config);

    if (config.filters == null) {
      throw new ValueError(
          'The `filters` configuration field is required by SeparableConv, ' +
          'but is unspecified.');
    }
    if (config.kernelInitializer != null || config.kernelRegularizer != null ||
        config.kernelConstraint != null) {
      throw new ValueError(
          'Fields kernelInitializer, kernelRegularizer and kernelConstraint ' +
          'are invalid for SeparableConv2D. Use depthwiseInitializer, ' +
          'depthwiseRegularizer, depthwiseConstraint, pointwiseInitializer, ' +
          'pointwiseRegularizer and pointwiseConstraint instead.');
    }
    if (config.padding != null && config.padding !== 'same' &&
        config.padding !== 'valid') {
      throw new ValueError(
          `SeparableConv${this.rank}D supports only padding modes: ` +
          `'same' and 'valid', but received ${JSON.stringify(config.padding)}`);
    }

    this.depthMultiplier =
        config.depthMultiplier == null ? 1 : config.depthMultiplier;
    this.depthwiseInitializer = getInitializer(
        config.depthwiseInitializer || this.DEFAULT_DEPTHWISE_INITIALIZER);
    this.depthwiseRegularizer = getRegularizer(config.depthwiseRegularizer);
    this.depthwiseConstraint = getConstraint(config.depthwiseConstraint);
    this.pointwiseInitializer = getInitializer(
        config.depthwiseInitializer || this.DEFAULT_POINTWISE_INITIALIZER);
    this.pointwiseRegularizer = getRegularizer(config.pointwiseRegularizer);
    this.pointwiseConstraint = getConstraint(config.pointwiseConstraint);
  }

  build(inputShape: Shape|Shape[]): void {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    if (inputShape.length < this.rank + 2) {
      throw new ValueError(
          `Inputs to SeparableConv${this.rank}D should have rank ` +
          `${this.rank + 2}, but received input shape: ` +
          `${JSON.stringify(inputShape)}`);
    }
    const channelAxis =
        this.dataFormat === 'channelsFirst' ? 1 : inputShape.length - 1;
    if (inputShape[channelAxis] == null || inputShape[channelAxis] < 0) {
      throw new ValueError(
          `The channel dimension of the inputs should be defined, ` +
          `but found ${JSON.stringify(inputShape[channelAxis])}`);
    }

    const inputDim = inputShape[channelAxis];
    const depthwiseKernelShape =
        this.kernelSize.concat([inputDim, this.depthMultiplier]);
    const pointwiseKernelShape = [];
    for (let i = 0; i < this.rank; ++i) {
      pointwiseKernelShape.push(1);
    }
    pointwiseKernelShape.push(inputDim * this.depthMultiplier, this.filters);

    const trainable = true;
    this.depthwiseKernel = this.addWeight(
        'depthwise_kernel', depthwiseKernelShape, DType.float32,
        this.depthwiseInitializer, this.depthwiseRegularizer, trainable,
        this.depthwiseConstraint);
    this.pointwiseKernel = this.addWeight(
        'pointwise_kernel', pointwiseKernelShape, DType.float32,
        this.pointwiseInitializer, this.pointwiseRegularizer, trainable,
        this.pointwiseConstraint);
    if (this.useBias) {
      this.bias = this.addWeight(
          'bias', [this.filters], DType.float32, this.biasInitializer,
          this.biasRegularizer, trainable, this.biasConstraint);
    } else {
      this.bias = null;
    }

    this.inputSpec =
        [new InputSpec({ndim: this.rank + 2, axes: {[channelAxis]: inputDim}})];
    this.built = true;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    inputs = generic_utils.getExactlyOneTensor(inputs);

    let output: Tensor;
    if (this.rank === 1) {
      throw new NotImplementedError(
          '1D separable convolution is not implemented yet.');
    } else if (this.rank === 2) {
      if (this.dataFormat === 'channelsFirst') {
        inputs = transpose(inputs, [0, 2, 3, 1]);  // NCHW -> NHWC.
      }

      output = separableConv2d(
          inputs as Tensor4D, this.depthwiseKernel.read() as Tensor4D,
          this.pointwiseKernel.read() as Tensor4D,
          this.strides as [number, number], this.padding as 'same' | 'valid',
          this.dilationRate as [number, number], 'NHWC');
    }

    if (this.useBias) {
      output = K.biasAdd(output, this.bias.read(), this.dataFormat);
    }
    if (this.activation != null) {
      output = this.activation(output);
    }

    if (this.dataFormat === 'channelsFirst') {
      output = transpose(output, [0, 3, 1, 2]);  // NHWC -> NCHW.
    }
    return output;
  }

  getConfig(): serialization.ConfigDict {
    const config = super.getConfig();
    delete config['rank'];
    delete config['kernelInitializer'];
    delete config['kernelRegularizer'];
    delete config['kernelConstraint'];
    config['depthwiseInitializer'] =
        serializeInitializer(this.depthwiseInitializer);
    config['pointwiseInitializer'] =
        serializeInitializer(this.pointwiseInitializer);
    config['depthwiseRegularizer'] =
        serializeRegularizer(this.depthwiseRegularizer);
    config['pointwiseRegularizer'] =
        serializeRegularizer(this.pointwiseRegularizer);
    config['depthwiseConstraint'] =
        serializeConstraint(this.depthwiseConstraint);
    config['pointwiseConstraint'] =
        serializeConstraint(this.pointwiseConstraint);
    return config;
  }
}

/**
 * Depthwise separable 2D convolution.
 *
 * Separable convolution consists of first performing
 * a depthwise spatial convolution
 * (which acts on each input channel separately)
 * followed by a pointwise convolution which mixes together the resulting
 * output channels. The `depthMultiplier` argument controls how many
 * output channels are generated per input channel in the depthwise step.
 *
 * Intuitively, separable convolutions can be understood as
 * a way to factorize a convolution kernel into two smaller kernels,
 * or as an extreme version of an Inception block.
 *
 * Input shape:
 *   4D tensor with shape:
 *     `[batch, channels, rows, cols]` if data_format='channelsFirst'
 *   or 4D tensor with shape:
 *     `[batch, rows, cols, channels]` if data_format='channelsLast'.
 *
 * Output shape:
 *   4D tensor with shape:
 *     `[batch, filters, newRows, newCols]` if data_format='channelsFirst'
 *   or 4D tensor with shape:
 *     `[batch, newRows, newCols, filters]` if data_format='channelsLast'.
 *     `rows` and `cols` values might have changed due to padding.
 */
export class SeparableConv2D extends SeparableConv {
  static className = 'SeparableConv2D';
  constructor(config?: SeparableConvLayerConfig) {
    super(2, config);
  }
}
serialization.SerializationMap.register(SeparableConv2D);

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
  static className = 'Conv1D';
  constructor(config: ConvLayerConfig) {
    super(1, config);
    this.inputSpec = [{ndim: 3}];
  }

  getConfig(): serialization.ConfigDict {
    const config = super.getConfig();
    delete config['rank'];
    delete config['dataFormat'];
    return config;
  }
}
serialization.SerializationMap.register(Conv1D);

export interface Cropping2DLayerConfig extends LayerConfig {
  /**
   * Dimension of the corpping along the width and the height.
   * - If integer: the same symmetric cropping
   *  is applied to width and height.
   * - If list of 2 integers:
   *   interpreted as two different
   *   symmetric cropping values for height and width:
   *   `[symmetric_height_crop, symmetric_width_crop]`.
   * - If a list of 2 list of 2 integers:
   *   interpreted as
   *   `[[top_crop, bottom_crop], [left_crop, right_crop]]`
   */
  cropping: number|[number, number]|[[number, number], [number, number]];

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
}

/**
 * Cropping layer for 2D input (e.g., image).
 *
 * This layer can crop an input
 * at the top, bottom, left and right side of an image tensor.
 *
 * Input shape:
 *   4D tensor with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, rows, cols, channels]`
 *   - If `data_format` is `"channels_first"`:
 *     `[batch, channels, rows, cols]`.
 *
 * Output shape:
 *   4D with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, croppedRows, croppedCols, channels]`
 *    - If `dataFormat` is `"channelsFirst"`:
 *     `[batch, channels, croppedRows, croppedCols]`.
 *
 * Examples
 * ```js
 *
 * const model = tf.sequential();
 * model.add(tf.layers.cropping2D({cropping:[[2, 2], [2, 2]],
 *                                inputShape: [128, 128, 3]}));
 * //now output shape is [batch, 124, 124, 3]
 * ```
 */
export class Cropping2D extends Layer {
  static className = 'Cropping2D';
  protected readonly cropping: [[number, number], [number, number]];
  protected readonly dataFormat: DataFormat;

  constructor(config: Cropping2DLayerConfig) {
    super(config);
    if (typeof config.cropping === 'number')
      this.cropping = [
        [config.cropping, config.cropping], [config.cropping, config.cropping]
      ];
    else if (typeof config.cropping[0] === 'number')
      this.cropping = [
        [config.cropping[0] as number, config.cropping[0] as number],
        [config.cropping[1] as number, config.cropping[1] as number]
      ];
    else
      this.cropping = config.cropping as [[number, number], [number, number]];
    this.dataFormat =
        config.dataFormat === undefined ? 'channelsLast' : config.dataFormat;
    this.inputSpec = [{ndim: 4}];
  }

  computeOutputShape(inputShape: Shape): Shape {
    if (this.dataFormat === 'channelsFirst')
      return [
        inputShape[0], inputShape[1],
        inputShape[2] - this.cropping[0][0] - this.cropping[0][1],
        inputShape[2] - this.cropping[1][0] - this.cropping[1][1]
      ];
    else
      return [
        inputShape[0],
        inputShape[1] - this.cropping[0][0] - this.cropping[0][1],
        inputShape[2] - this.cropping[1][0] - this.cropping[1][1], inputShape[3]
      ];
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    inputs = generic_utils.getExactlyOneTensor(inputs);

    if (this.dataFormat === 'channelsLast') {
      const hSliced = K.sliceAlongAxis(
          inputs, this.cropping[0][0],
          inputs.shape[1] - this.cropping[0][0] - this.cropping[0][1], 2);
      return K.sliceAlongAxis(
          hSliced, this.cropping[1][0],
          inputs.shape[2] - this.cropping[1][1] - this.cropping[1][0], 3);
    } else {
      const hSliced = K.sliceAlongAxis(
          inputs, this.cropping[0][0],
          inputs.shape[2] - this.cropping[0][0] - this.cropping[0][1], 3);
      return K.sliceAlongAxis(
          hSliced, this.cropping[1][0],
          inputs.shape[3] - this.cropping[1][1] - this.cropping[1][0], 4);
    }
  }

  getConfig(): serialization.ConfigDict {
    const config = {cropping: this.cropping, dataFormat: this.dataFormat};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.SerializationMap.register(Cropping2D);
