import {serialization, Tensor, tidy} from '@tensorflow/tfjs-core';

import {Activation, getActivation, serializeActivation} from '../activations';
import {checkDataFormat, checkPaddingMode} from '../common';
import {Constraint, getConstraint, serializeConstraint} from '../constraints';
import {getInitializer, Initializer, InitializerIdentifier, serializeInitializer} from '../initializers';
import {ActivationIdentifier} from '../keras_format/activation_config';
import {DataFormat, PaddingMode, Shape} from '../keras_format/common';
import {getRegularizer, Regularizer, serializeRegularizer} from '../regularizers';
import {Kwargs} from '../types';
import {normalizeArray} from '../utils/conv_utils';
import {assertPositiveInteger} from '../utils/generic_utils';
import * as math_utils from '../utils/math_utils';
import {LayerVariable} from '../variables';

import {RNNCell, SimpleRNNCellLayerArgs} from './recurrent';

export declare interface ConvLSTM2DCellArgs extends SimpleRNNCellLayerArgs {
  /**
   * The dimensionality of the output space (i.e. the number of filters in the
   * convolution).
   */
  filters: number;

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
   * Activation function to use for the recurrent step.
   *
   * Defaults to hard sigmoid (`hardSigmoid`).
   *
   * If `null`, no activation is applied.
   */
  recurrentActivation?: ActivationIdentifier;

  /**
   * If `true`, add 1 to the bias of the forget gate at initialization.
   * Setting it to `true` will also force `biasInitializer = 'zeros'`.
   * This is recommended in
   * [Jozefowicz et
   * al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
   */
  unitForgetBias?: boolean;

  /**
   * Implementation mode, either 1 or 2.
   *
   * Mode 1 will structure its operations as a larger number of
   *   smaller dot products and additions.
   *
   * Mode 2 will batch them into fewer, larger operations. These modes will
   * have different performance profiles on different hardware and
   * for different applications.
   *
   * Note: For superior performance, TensorFlow.js always uses implementation
   * 2, regardless of the actual value of this configuration field.
   */
  implementation?: number;
}

export class ConvLSTM2DCell extends RNNCell {
  /** @nocollapse */
  static className = 'ConvLSTM2DCell';
  readonly filters: number;
  readonly kernelSize: number[];
  readonly strides: number[];
  readonly padding: PaddingMode;
  readonly dataFormat: DataFormat;
  readonly dilationRate: number[];

  readonly activation: Activation;
  readonly recurrentActivation: Activation;
  readonly useBias: boolean;

  readonly kernelInitializer: Initializer;
  readonly recurrentInitializer: Initializer;
  readonly biasInitializer: Initializer;
  readonly unitForgetBias: boolean;

  readonly kernelConstraint: Constraint;
  readonly recurrentConstraint: Constraint;
  readonly biasConstraint: Constraint;

  readonly kernelRegularizer: Regularizer;
  readonly recurrentRegularizer: Regularizer;
  readonly biasRegularizer: Regularizer;

  readonly dropout: number;
  readonly recurrentDropout: number;

  readonly stateSize: number[];
  readonly implementation: number;

  readonly DEFAULT_ACTIVATION = 'tanh';
  readonly DEFAULT_RECURRENT_ACTIVATION: ActivationIdentifier = 'hardSigmoid';

  readonly DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
  readonly DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';
  readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier = 'zeros';

  kernel: LayerVariable;
  recurrentKernel: LayerVariable;
  bias: LayerVariable;

  constructor(args: ConvLSTM2DCellArgs) {
    super(args);

    this.filters = args.filters;
    assertPositiveInteger(this.filters, 'filters');
    this.kernelSize = normalizeArray(args.kernelSize, 2, 'kernelSize');
    this.strides =
        normalizeArray(args.strides == null ? 1 : args.strides, 2, 'strides');
    this.padding = args.padding == null ? 'valid' : args.padding;
    checkPaddingMode(this.padding);
    this.dataFormat =
        args.dataFormat == null ? 'channelsLast' : args.dataFormat;
    checkDataFormat(this.dataFormat);

    this.activation = getActivation(
        args.activation === undefined ? this.DEFAULT_ACTIVATION :
                                        args.activation);
    this.recurrentActivation = getActivation(
        args.recurrentActivation === undefined ?
            this.DEFAULT_RECURRENT_ACTIVATION :
            args.recurrentActivation);
    this.useBias = args.useBias == null ? true : args.useBias;

    this.kernelInitializer = getInitializer(
        args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.recurrentInitializer = getInitializer(
        args.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);

    this.biasInitializer =
        getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
    this.unitForgetBias = args.unitForgetBias;

    this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
    this.recurrentRegularizer = getRegularizer(args.recurrentRegularizer);
    this.biasRegularizer = getRegularizer(args.biasRegularizer);

    this.kernelConstraint = getConstraint(args.kernelConstraint);
    this.recurrentConstraint = getConstraint(args.recurrentConstraint);
    this.biasConstraint = getConstraint(args.biasConstraint);

    this.dropout = math_utils.min(
        [1, math_utils.max([0, args.dropout == null ? 0 : args.dropout])]);
    this.recurrentDropout = math_utils.min([
      1,
      math_utils.max(
          [0, args.recurrentDropout == null ? 0 : args.recurrentDropout])
    ]);
    this.implementation = args.implementation;
    // this.stateSize = [this.units, this.units];
    // this.dropoutMask = null;
    // this.recurrentDropoutMask = null;
  }

  public build(inputShape: Shape|Shape[]): void {
    this.built = true;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      return inputs;
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      filters: this.filters,
      kernelSize: this.kernelSize,
      padding: this.padding,
      dataFormat: this.dataFormat,
      dilationRate: this.dilationRate,
      activation: serializeActivation(this.activation),
      recurrentActivation: serializeActivation(this.recurrentActivation),
      useBias: this.useBias,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      recurrentInitializer: serializeInitializer(this.recurrentInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      unitForgetBias: this.unitForgetBias,
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint),
      recurrentConstraint: serializeConstraint(this.recurrentConstraint),
      biasConstraint: serializeConstraint(this.biasConstraint),
      dropout: this.dropout,
      recurrentDropout: this.recurrentDropout,
      implementation: this.implementation,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}

serialization.registerClass(ConvLSTM2DCell);
