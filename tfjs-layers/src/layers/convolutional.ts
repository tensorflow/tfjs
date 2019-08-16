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

import * as tfc from '@tensorflow/tfjs-core';
import {serialization, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, tidy} from '@tensorflow/tfjs-core';

import {Activation, getActivation, serializeActivation} from '../activations';
import {imageDataFormat} from '../backend/common';
import * as K from '../backend/tfjs_backend';
import {checkDataFormat, checkPaddingMode} from '../common';
import {Constraint, ConstraintIdentifier, getConstraint, serializeConstraint} from '../constraints';
import {InputSpec, Layer, LayerArgs} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, serializeInitializer} from '../initializers';
import {ActivationIdentifier} from '../keras_format/activation_config';
import {DataFormat, PaddingMode, Shape} from '../keras_format/common';
import {getRegularizer, Regularizer, RegularizerIdentifier, serializeRegularizer} from '../regularizers';
import {Kwargs} from '../types';
import {convOutputLength, deconvLength, normalizeArray} from '../utils/conv_utils';
import * as generic_utils from '../utils/generic_utils';
import {getExactlyOneShape, getExactlyOneTensor} from '../utils/types_utils';
import {LayerVariable} from '../variables';

/**
 * Transpose and cast the input before the conv2d.
 * @param x Input image tensor.
 * @param dataFormat
 */
export function preprocessConv2DInput(
    x: Tensor, dataFormat: DataFormat): Tensor {
  // TODO(cais): Cast type to float32 if not.
  return tidy(() => {
    checkDataFormat(dataFormat);
    if (dataFormat === 'channelsFirst') {
      return tfc.transpose(x, [0, 2, 3, 1]);  // NCHW -> NHWC.
    } else {
      return x;
    }
  });
}

/**
 * Transpose and cast the input before the conv3d.
 * @param x Input image tensor.
 * @param dataFormat
 */
export function preprocessConv3DInput(
    x: Tensor, dataFormat: DataFormat): Tensor {
  return tidy(() => {
    checkDataFormat(dataFormat);
    if (dataFormat === 'channelsFirst') {
      return tfc.transpose(x, [0, 2, 3, 4, 1]);  // NCDHW -> NDHWC.
    } else {
      return x;
    }
  });
}

/**
 * 1D-convolution with bias added.
 *
 * Porting Note: This function does not exist in the Python Keras backend.
 *   It is exactly the same as `conv2d`, except the added `bias`.
 *
 * @param x Input tensor, rank-3, of shape `[batchSize, width, inChannels]`.
 * @param kernel Kernel, rank-3, of shape `[filterWidth, inDepth, outDepth]`.
 * @param bias Bias, rank-3, of shape `[outDepth]`.
 * @param strides
 * @param padding Padding mode.
 * @param dataFormat Data format.
 * @param dilationRate
 * @returns The result of the 1D convolution.
 * @throws ValueError, if `x`, `kernel` or `bias` is not of the correct rank.
 */
export function conv1dWithBias(
    x: Tensor, kernel: Tensor, bias: Tensor, strides = 1, padding = 'valid',
    dataFormat?: DataFormat, dilationRate = 1): Tensor {
  return tidy(() => {
    if (dataFormat == null) {
      dataFormat = imageDataFormat();
    }
    checkDataFormat(dataFormat);
    // Check the ranks of x, kernel and bias.
    if (x.shape.length !== 3) {
      throw new ValueError(
          `The input of a conv1dWithBias operation should be 3, but is ` +
          `${x.shape.length} instead.`);
    }
    if (kernel.shape.length !== 3) {
      throw new ValueError(
          `The kernel for a conv1dWithBias operation should be 3, but is ` +
          `${kernel.shape.length} instead`);
    }
    if (bias != null && bias.shape.length !== 1) {
      throw new ValueError(
          `The bias for a conv1dWithBias operation should be 1, but is ` +
          `${kernel.shape.length} instead`);
    }
    // TODO(cais): Support CAUSAL padding mode.
    if (dataFormat === 'channelsFirst') {
      x = tfc.transpose(x, [0, 2, 1]);  // NCW -> NWC.
    }
    if (padding === 'causal') {
      throw new NotImplementedError(
          'The support for CAUSAL padding mode in conv1dWithBias is not ' +
          'implemented yet.');
    }
    let y: Tensor = tfc.conv1d(
        x as Tensor2D | Tensor3D, kernel as Tensor3D, strides,
        padding === 'same' ? 'same' : 'valid', 'NWC', dilationRate);
    if (bias != null) {
      y = K.biasAdd(y, bias);
    }
    return y;
  });
}

/**
 * 1D-convolution.
 *
 * @param x Input tensor, rank-3, of shape `[batchSize, width, inChannels]`.
 * @param kernel Kernel, rank-3, of shape `[filterWidth, inDepth, outDepth]`.s
 * @param strides
 * @param padding Padding mode.
 * @param dataFormat Data format.
 * @param dilationRate
 * @returns The result of the 1D convolution.
 * @throws ValueError, if `x`, `kernel` or `bias` is not of the correct rank.
 */
export function conv1d(
    x: Tensor, kernel: Tensor, strides = 1, padding = 'valid',
    dataFormat?: DataFormat, dilationRate = 1): Tensor {
  return tidy(() => {
    checkDataFormat(dataFormat);
    return conv1dWithBias(
        x, kernel, null, strides, padding, dataFormat, dilationRate);
  });
}

/**
 * 2D Convolution
 * @param x
 * @param kernel kernel of the convolution.
 * @param strides strides array.
 * @param padding padding mode. Default to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param dilationRate dilation rate array.
 * @returns Result of the 2D pooling.
 */
export function conv2d(
    x: Tensor, kernel: Tensor, strides = [1, 1], padding = 'valid',
    dataFormat?: DataFormat, dilationRate?: [number, number]): Tensor {
  return tidy(() => {
    checkDataFormat(dataFormat);
    return conv2dWithBias(
        x, kernel, null, strides, padding, dataFormat, dilationRate);
  });
}

/**
 * 2D Convolution with an added bias.
 * Note: This function does not exist in the Python Keras Backend. This function
 * is exactly the same as `conv2d`, except the added `bias`.
 */
export function conv2dWithBias(
    x: Tensor, kernel: Tensor, bias: Tensor, strides = [1, 1],
    padding = 'valid', dataFormat?: DataFormat,
    dilationRate?: [number, number]): Tensor {
  return tidy(() => {
    if (dataFormat == null) {
      dataFormat = imageDataFormat();
    }
    checkDataFormat(dataFormat);
    if (x.rank !== 3 && x.rank !== 4) {
      throw new ValueError(
          `conv2dWithBias expects input to be of rank 3 or 4, but received ` +
          `${x.rank}.`);
    }
    if (kernel.rank !== 3 && kernel.rank !== 4) {
      throw new ValueError(
          `conv2dWithBias expects kernel to be of rank 3 or 4, but received ` +
          `${x.rank}.`);
    }
    let y = preprocessConv2DInput(x, dataFormat);
    if (padding === 'causal') {
      throw new NotImplementedError(
          'The support for CAUSAL padding mode in conv1dWithBias is not ' +
          'implemented yet.');
    }
    y = tfc.conv2d(
        y as Tensor3D | Tensor4D, kernel as Tensor4D,
        strides as [number, number], padding === 'same' ? 'same' : 'valid',
        'NHWC', dilationRate);
    if (bias != null) {
      y = K.biasAdd(y, bias as Tensor1D);
    }
    if (dataFormat === 'channelsFirst') {
      y = tfc.transpose(y, [0, 3, 1, 2]);
    }
    return y;
  });
}

/**
 * 3D Convolution.
 * @param x
 * @param kernel kernel of the convolution.
 * @param strides strides array.
 * @param padding padding mode. Default to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param dilationRate dilation rate array.
 * @returns Result of the 3D convolution.
 */
export function conv3d(
    x: Tensor, kernel: Tensor, strides = [1, 1, 1], padding = 'valid',
    dataFormat?: DataFormat, dilationRate?: [number, number, number]): Tensor {
  return tidy(() => {
    checkDataFormat(dataFormat);
    return conv3dWithBias(
        x, kernel, null, strides, padding, dataFormat, dilationRate);
  });
}

/**
 * 3D Convolution with an added bias.
 * Note: This function does not exist in the Python Keras Backend. This function
 * is exactly the same as `conv3d`, except the added `bias`.
 */
export function conv3dWithBias(
    x: Tensor, kernel: Tensor, bias: Tensor, strides = [1, 1, 1],
    padding = 'valid', dataFormat?: DataFormat,
    dilationRate?: [number, number, number]): Tensor {
  return tidy(() => {
    if (dataFormat == null) {
      dataFormat = imageDataFormat();
    }
    checkDataFormat(dataFormat);
    if (x.rank !== 4 && x.rank !== 5) {
      throw new ValueError(
          `conv3dWithBias expects input to be of rank 4 or 5, but received ` +
          `${x.rank}.`);
    }
    if (kernel.rank !== 4 && kernel.rank !== 5) {
      throw new ValueError(
          `conv3dWithBias expects kernel to be of rank 4 or 5, but received ` +
          `${x.rank}.`);
    }
    let y = preprocessConv3DInput(x, dataFormat);
    if (padding === 'causal') {
      throw new NotImplementedError(
          'The support for CAUSAL padding mode in conv3dWithBias is not ' +
          'implemented yet.');
    }
    y = tfc.conv3d(
        y as Tensor4D | tfc.Tensor<tfc.Rank.R5>,
        kernel as tfc.Tensor<tfc.Rank.R5>, strides as [number, number, number],
        padding === 'same' ? 'same' : 'valid', 'NDHWC', dilationRate);
    if (bias != null) {
      y = K.biasAdd(y, bias as Tensor1D);
    }
    if (dataFormat === 'channelsFirst') {
      y = tfc.transpose(y, [0, 4, 1, 2, 3]);
    }
    return y;
  });
}


/**
 * Base LayerConfig for depthwise and non-depthwise convolutional layers.
 */
export declare interface BaseConvLayerArgs extends LayerArgs {
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
   * Should be an integer or array of two or three integers.
   *
   * Currently, specifying any `dilationRate` value != 1 is incompatible with
   * specifying any `strides` value != 1.
   */
  dilationRate?: number|[number]|[number, number]|[number, number, number];

  /**
   * Activation function of the layer.
   *
   * If you don't specify the activation, none is applied.
   */
  activation?: ActivationIdentifier;

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
 * Applies to non-depthwise convolution of all ranks (e.g, Conv1D, Conv2D,
 * Conv3D).
 */
export declare interface ConvLayerArgs extends BaseConvLayerArgs {
  /**
   * The dimensionality of the output space (i.e. the number of filters in the
   * convolution).
   */
  filters: number;
}

/**
 * Abstract convolution layer.
 */
export abstract class BaseConv extends Layer {
  protected readonly rank: number;
  protected readonly kernelSize: number[];
  protected readonly strides: number[];
  protected readonly padding: PaddingMode;
  protected readonly dataFormat: DataFormat;
  protected readonly activation: Activation;
  protected readonly useBias: boolean;
  protected readonly dilationRate: number[];

  // Bias-related members are here because all convolution subclasses use the
  // same configuration parmeters to control bias.  Kernel-related members
  // are in subclass `Conv` because some subclasses use different parameters to
  // control kernel properties, for instance, `DepthwiseConv2D` uses
  // `depthwiseInitializer` instead of `kernelInitializer`.
  protected readonly biasInitializer?: Initializer;
  protected readonly biasConstraint?: Constraint;
  protected readonly biasRegularizer?: Regularizer;

  protected bias: LayerVariable = null;

  readonly DEFAULT_KERNEL_INITIALIZER: InitializerIdentifier = 'glorotNormal';
  readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier = 'zeros';

  constructor(rank: number, args: BaseConvLayerArgs) {
    super(args as LayerArgs);
    BaseConv.verifyArgs(args);
    this.rank = rank;
    generic_utils.assertPositiveInteger(this.rank, 'rank');
    if (this.rank !== 1 && this.rank !== 2 && this.rank !== 3) {
      throw new NotImplementedError(
          `Convolution layer for rank other than 1, 2, or 3 (${
              this.rank}) is ` +
          `not implemented yet.`);
    }
    this.kernelSize = normalizeArray(args.kernelSize, rank, 'kernelSize');
    this.strides = normalizeArray(
        args.strides == null ? 1 : args.strides, rank, 'strides');
    this.padding = args.padding == null ? 'valid' : args.padding;
    checkPaddingMode(this.padding);
    this.dataFormat =
        args.dataFormat == null ? 'channelsLast' : args.dataFormat;
    checkDataFormat(this.dataFormat);
    this.activation = getActivation(args.activation);
    this.useBias = args.useBias == null ? true : args.useBias;
    this.biasInitializer =
        getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
    this.biasConstraint = getConstraint(args.biasConstraint);
    this.biasRegularizer = getRegularizer(args.biasRegularizer);
    this.activityRegularizer = getRegularizer(args.activityRegularizer);
    this.dilationRate = normalizeArray(
        args.dilationRate == null ? 1 : args.dilationRate, rank,
        'dilationRate');
    if (this.rank === 1 &&
        (Array.isArray(this.dilationRate) &&
         (this.dilationRate as number[]).length !== 1)) {
      throw new ValueError(
          `dilationRate must be a number or an array of a single number ` +
          `for 1D convolution, but received ` +
          `${JSON.stringify(this.dilationRate)}`);
    } else if (this.rank === 2) {
      if (typeof this.dilationRate === 'number') {
        this.dilationRate = [this.dilationRate, this.dilationRate];
      } else if (this.dilationRate.length !== 2) {
        throw new ValueError(
            `dilationRate must be a number or array of two numbers for 2D ` +
            `convolution, but received ${JSON.stringify(this.dilationRate)}`);
      }
    } else if (this.rank === 3) {
      if (typeof this.dilationRate === 'number') {
        this.dilationRate =
            [this.dilationRate, this.dilationRate, this.dilationRate];
      } else if (this.dilationRate.length !== 3) {
        throw new ValueError(
            `dilationRate must be a number or array of three numbers for 3D ` +
            `convolution, but received ${JSON.stringify(this.dilationRate)}`);
      }
    }
  }

  protected static verifyArgs(args: BaseConvLayerArgs) {
    // Check config.kernelSize type and shape.
    generic_utils.assert(
        'kernelSize' in args, `required key 'kernelSize' not in config`);
    if (typeof args.kernelSize !== 'number' &&
        !generic_utils.checkArrayTypeAndLength(args.kernelSize, 'number', 1, 3))
      throw new ValueError(
          `BaseConv expects config.kernelSize to be number or number[] with ` +
          `length 1, 2, or 3, but received ${
              JSON.stringify(args.kernelSize)}.`);
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      kernelSize: this.kernelSize,
      strides: this.strides,
      padding: this.padding,
      dataFormat: this.dataFormat,
      dilationRate: this.dilationRate,
      activation: serializeActivation(this.activation),
      useBias: this.useBias,
      biasInitializer: serializeInitializer(this.biasInitializer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      biasConstraint: serializeConstraint(this.biasConstraint)
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}

/**
 * Abstract nD convolution layer.  Ancestor of convolution layers which reduce
 * across channels, i.e., Conv1D and Conv2D, but not DepthwiseConv2D.
 */
export abstract class Conv extends BaseConv {
  protected readonly filters: number;

  protected kernel: LayerVariable = null;

  // Bias-related properties are stored in the superclass `BaseConv` because all
  // convolution subclasses use the same configuration parameters to control
  // bias. Kernel-related properties are defined here rather than in the
  // superclass because some convolution subclasses use different names and
  // configuration parameters for their internal kernel state.
  protected readonly kernelInitializer?: Initializer;
  protected readonly kernelConstraint?: Constraint;
  protected readonly kernelRegularizer?: Regularizer;

  constructor(rank: number, args: ConvLayerArgs) {
    super(rank, args as BaseConvLayerArgs);
    Conv.verifyArgs(args);
    this.filters = args.filters;
    generic_utils.assertPositiveInteger(this.filters, 'filters');
    this.kernelInitializer = getInitializer(
        args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.kernelConstraint = getConstraint(args.kernelConstraint);
    this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
  }

  build(inputShape: Shape|Shape[]): void {
    inputShape = getExactlyOneShape(inputShape);
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
    return tidy(() => {
      inputs = getExactlyOneTensor(inputs);
      let outputs: Tensor;
      const biasValue = this.bias == null ? null : this.bias.read();

      if (this.rank === 1) {
        outputs = conv1dWithBias(
            inputs, this.kernel.read(), biasValue, this.strides[0],
            this.padding, this.dataFormat, this.dilationRate[0]);
      } else if (this.rank === 2) {
        // TODO(cais): Move up to constructor.
        outputs = conv2dWithBias(
            inputs, this.kernel.read(), biasValue, this.strides, this.padding,
            this.dataFormat, this.dilationRate as [number, number]);
      } else if (this.rank === 3) {
        outputs = conv3dWithBias(
            inputs, this.kernel.read(), biasValue, this.strides, this.padding,
            this.dataFormat, this.dilationRate as [number, number, number]);
      } else {
        throw new NotImplementedError(
            'convolutions greater than 3D are not implemented yet.');
      }

      if (this.activation != null) {
        outputs = this.activation.apply(outputs);
      }
      return outputs;
    });
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
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
    const config = {
      filters: this.filters,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint)
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  protected static verifyArgs(args: ConvLayerArgs) {
    // Check config.filters type, shape, and value.
    if (!('filters' in args) || typeof args.filters !== 'number' ||
        args.filters < 1) {
      throw new ValueError(
          `Convolution layer expected config.filters to be a 'number' > 0 ` +
          `but got ${JSON.stringify(args.filters)}`);
    }
  }
}

export class Conv2D extends Conv {
  /** @nocollapse */
  static className = 'Conv2D';
  constructor(args: ConvLayerArgs) {
    super(2, args);
    Conv2D.verifyArgs(args);
  }

  getConfig(): serialization.ConfigDict {
    const config = super.getConfig();
    delete config['rank'];
    return config;
  }

  protected static verifyArgs(args: ConvLayerArgs) {
    // config.kernelSize must be a number or array of numbers.
    if ((typeof args.kernelSize !== 'number') &&
        !generic_utils.checkArrayTypeAndLength(args.kernelSize, 'number', 1, 2))
      throw new ValueError(
          `Conv2D expects config.kernelSize to be number or number[] with ` +
          `length 1 or 2, but received ${JSON.stringify(args.kernelSize)}.`);
  }
}
serialization.registerClass(Conv2D);

export class Conv3D extends Conv {
  /** @nocollapse */
  static className = 'Conv3D';
  constructor(args: ConvLayerArgs) {
    super(3, args);
    Conv3D.verifyArgs(args);
  }

  getConfig(): serialization.ConfigDict {
    const config = super.getConfig();
    delete config['rank'];
    return config;
  }

  protected static verifyArgs(args: ConvLayerArgs) {
    // config.kernelSize must be a number or array of numbers.
    if (typeof args.kernelSize !== 'number') {
      if (!(Array.isArray(args.kernelSize) &&
            (args.kernelSize.length === 1 || args.kernelSize.length === 3)))
        throw new ValueError(
            `Conv3D expects config.kernelSize to be number or` +
            ` [number, number, number], but received ${
                JSON.stringify(args.kernelSize)}.`);
    }
  }
}
serialization.registerClass(Conv3D);

export class Conv2DTranspose extends Conv2D {
  /** @nocollapse */
  static className = 'Conv2DTranspose';
  inputSpec: InputSpec[];

  constructor(args: ConvLayerArgs) {
    super(args);
    this.inputSpec = [new InputSpec({ndim: 4})];

    if (this.padding !== 'same' && this.padding !== 'valid') {
      throw new ValueError(
          `Conv2DTranspose currently supports only padding modes 'same' ` +
          `and 'valid', but received padding mode ${this.padding}`);
    }
  }

  build(inputShape: Shape|Shape[]): void {
    inputShape = getExactlyOneShape(inputShape);

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
        'kernel', kernelShape, 'float32', this.kernelInitializer,
        this.kernelRegularizer, true, this.kernelConstraint);
    if (this.useBias) {
      this.bias = this.addWeight(
          'bias', [this.filters], 'float32', this.biasInitializer,
          this.biasRegularizer, true, this.biasConstraint);
    }

    // Set input spec.
    this.inputSpec =
        [new InputSpec({ndim: 4, axes: {[channelAxis]: inputDim}})];
    this.built = true;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tfc.tidy(() => {
      let input = getExactlyOneTensor(inputs);
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

      // Porting Note: We don't branch based on `this.dataFormat` here,
      // because
      //   the tjfs-core function `conv2dTranspose` called below always
      //   assumes channelsLast.
      const outputShape: [number, number, number, number] =
          [batchSize, outHeight, outWidth, this.filters];

      if (this.dataFormat !== 'channelsLast') {
        input = tfc.transpose(input, [0, 2, 3, 1]);
      }
      let outputs = tfc.conv2dTranspose(
          input as Tensor4D, this.kernel.read() as Tensor4D, outputShape,
          this.strides as [number, number], this.padding as 'same' | 'valid');
      if (this.dataFormat !== 'channelsLast') {
        outputs = tfc.transpose(outputs, [0, 3, 1, 2]) as Tensor4D;
      }

      if (this.bias != null) {
        outputs =
            K.biasAdd(outputs, this.bias.read(), this.dataFormat) as Tensor4D;
      }
      if (this.activation != null) {
        outputs = this.activation.apply(outputs) as Tensor4D;
      }
      return outputs;
    });
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
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
serialization.registerClass(Conv2DTranspose);

export declare interface SeparableConvLayerArgs extends ConvLayerArgs {
  /**
   * The number of depthwise convolution output channels for each input
   * channel.
   * The total number of depthwise convolution output channels will be equal
   * to `filtersIn * depthMultiplier`. Default: 1.
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
  /** @nocollapse */
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

  constructor(rank: number, config?: SeparableConvLayerArgs) {
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
    inputShape = getExactlyOneShape(inputShape);
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
        'depthwise_kernel', depthwiseKernelShape, 'float32',
        this.depthwiseInitializer, this.depthwiseRegularizer, trainable,
        this.depthwiseConstraint);
    this.pointwiseKernel = this.addWeight(
        'pointwise_kernel', pointwiseKernelShape, 'float32',
        this.pointwiseInitializer, this.pointwiseRegularizer, trainable,
        this.pointwiseConstraint);
    if (this.useBias) {
      this.bias = this.addWeight(
          'bias', [this.filters], 'float32', this.biasInitializer,
          this.biasRegularizer, trainable, this.biasConstraint);
    } else {
      this.bias = null;
    }

    this.inputSpec =
        [new InputSpec({ndim: this.rank + 2, axes: {[channelAxis]: inputDim}})];
    this.built = true;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      inputs = getExactlyOneTensor(inputs);

      let output: Tensor;
      if (this.rank === 1) {
        throw new NotImplementedError(
            '1D separable convolution is not implemented yet.');
      } else if (this.rank === 2) {
        if (this.dataFormat === 'channelsFirst') {
          inputs = tfc.transpose(inputs, [0, 2, 3, 1]);  // NCHW -> NHWC.
        }

        output = tfc.separableConv2d(
            inputs as Tensor4D, this.depthwiseKernel.read() as Tensor4D,
            this.pointwiseKernel.read() as Tensor4D,
            this.strides as [number, number], this.padding as 'same' | 'valid',
            this.dilationRate as [number, number], 'NHWC');
      }

      if (this.useBias) {
        output = K.biasAdd(output, this.bias.read(), this.dataFormat);
      }
      if (this.activation != null) {
        output = this.activation.apply(output);
      }

      if (this.dataFormat === 'channelsFirst') {
        output = tfc.transpose(output, [0, 3, 1, 2]);  // NHWC -> NCHW.
      }
      return output;
    });
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

export class SeparableConv2D extends SeparableConv {
  /** @nocollapse */
  static className = 'SeparableConv2D';
  constructor(args?: SeparableConvLayerArgs) {
    super(2, args);
  }
}
serialization.registerClass(SeparableConv2D);

export class Conv1D extends Conv {
  /** @nocollapse */
  static className = 'Conv1D';
  constructor(args: ConvLayerArgs) {
    super(1, args);
    Conv1D.verifyArgs(args);
    this.inputSpec = [{ndim: 3}];
  }

  getConfig(): serialization.ConfigDict {
    const config = super.getConfig();
    delete config['rank'];
    delete config['dataFormat'];
    return config;
  }

  protected static verifyArgs(args: ConvLayerArgs) {
    // config.kernelSize must be a number or array of numbers.
    if (typeof args.kernelSize !== 'number' &&
        !generic_utils.checkArrayTypeAndLength(args.kernelSize, 'number', 1, 1))
      throw new ValueError(
          `Conv1D expects config.kernelSize to be number or number[] with ` +
          `length 1, but received ${JSON.stringify(args.kernelSize)}.`);
  }
}
serialization.registerClass(Conv1D);

export declare interface Cropping2DLayerArgs extends LayerArgs {
  /**
   * Dimension of the cropping along the width and the height.
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

export class Cropping2D extends Layer {
  /** @nocollapse */
  static className = 'Cropping2D';
  protected readonly cropping: [[number, number], [number, number]];
  protected readonly dataFormat: DataFormat;

  constructor(args: Cropping2DLayerArgs) {
    super(args);
    if (typeof args.cropping === 'number')
      this.cropping =
          [[args.cropping, args.cropping], [args.cropping, args.cropping]];
    else if (typeof args.cropping[0] === 'number')
      this.cropping = [
        [args.cropping[0] as number, args.cropping[0] as number],
        [args.cropping[1] as number, args.cropping[1] as number]
      ];
    else
      this.cropping = args.cropping as [[number, number], [number, number]];
    this.dataFormat =
        args.dataFormat === undefined ? 'channelsLast' : args.dataFormat;
    this.inputSpec = [{ndim: 4}];
  }

  computeOutputShape(inputShape: Shape): Shape {
    if (this.dataFormat === 'channelsFirst')
      return [
        inputShape[0], inputShape[1],
        inputShape[2] - this.cropping[0][0] - this.cropping[0][1],
        inputShape[3] - this.cropping[1][0] - this.cropping[1][1]
      ];
    else
      return [
        inputShape[0],
        inputShape[1] - this.cropping[0][0] - this.cropping[0][1],
        inputShape[2] - this.cropping[1][0] - this.cropping[1][1], inputShape[3]
      ];
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      inputs = getExactlyOneTensor(inputs);

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
    });
  }

  getConfig(): serialization.ConfigDict {
    const config = {cropping: this.cropping, dataFormat: this.dataFormat};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(Cropping2D);

export declare interface UpSampling2DLayerArgs extends LayerArgs {
  /**
   * The upsampling factors for rows and columns.
   *
   * Defaults to `[2, 2]`.
   */
  size?: number[];
  /**
   * Format of the data, which determines the ordering of the dimensions in
   * the inputs.
   *
   * `"channelsLast"` corresponds to inputs with shape
   *   `[batch, ..., channels]`
   *
   *  `"channelsFirst"` corresponds to inputs with shape `[batch, channels,
   * ...]`.
   *
   * Defaults to `"channelsLast"`.
   */
  dataFormat?: DataFormat;
}

export class UpSampling2D extends Layer {
  /** @nocollapse */
  static className = 'UpSampling2D';
  protected readonly DEFAULT_SIZE = [2, 2];
  protected readonly size: number[];
  protected readonly dataFormat: DataFormat;

  constructor(args: UpSampling2DLayerArgs) {
    super(args);
    this.inputSpec = [{ndim: 4}];
    this.size = args.size == null ? this.DEFAULT_SIZE : args.size;
    this.dataFormat =
        args.dataFormat == null ? 'channelsLast' : args.dataFormat;
  }

  computeOutputShape(inputShape: Shape): Shape {
    if (this.dataFormat === 'channelsFirst') {
      const height =
          inputShape[2] == null ? null : this.size[0] * inputShape[2];
      const width = inputShape[3] == null ? null : this.size[1] * inputShape[3];
      return [inputShape[0], inputShape[1], height, width];
    } else {
      const height =
          inputShape[1] == null ? null : this.size[0] * inputShape[1];
      const width = inputShape[2] == null ? null : this.size[1] * inputShape[2];
      return [inputShape[0], height, width, inputShape[3]];
    }
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tfc.tidy(() => {
      let input = getExactlyOneTensor(inputs) as Tensor4D;
      const inputShape = input.shape;

      if (this.dataFormat === 'channelsFirst') {
        input = tfc.transpose(input, [0, 2, 3, 1]);
        const height = this.size[0] * inputShape[2];
        const width = this.size[1] * inputShape[3];
        const resized = input.resizeNearestNeighbor([height, width]);
        return tfc.transpose(resized, [0, 3, 1, 2]);
      } else {
        const height = this.size[0] * inputShape[1];
        const width = this.size[1] * inputShape[2];
        return input.resizeNearestNeighbor([height, width]);
      }
    });
  }

  getConfig(): serialization.ConfigDict {
    const config = {size: this.size, dataFormat: this.dataFormat};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(UpSampling2D);
