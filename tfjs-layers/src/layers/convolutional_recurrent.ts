import * as tfc from '@tensorflow/tfjs-core';

import {serializeActivation} from '../activations';
import * as K from '../backend/tfjs_backend';
import {checkDataFormat, checkPaddingMode} from '../common';
import {serializeConstraint} from '../constraints';
import {InputSpec} from '../engine/topology';
import {ValueError} from '../errors';
import {Initializer, serializeInitializer} from '../initializers';
import {DataFormat, DataType, PaddingMode, Shape} from '../keras_format/common';
import {serializeRegularizer} from '../regularizers';
import {Kwargs} from '../types';
import {convOutputLength, normalizeArray} from '../utils/conv_utils';
import {assertPositiveInteger} from '../utils/generic_utils';
import {getExactlyOneShape} from '../utils/types_utils';

import {LSTMCell, LSTMCellLayerArgs, LSTMLayerArgs, RNN, RNNLayerArgs,} from './recurrent';

export declare interface ConvLSTM2DCellArgs extends
    Omit<LSTMCellLayerArgs, 'units'> {
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
  dilationRate?: number|[number]|[number, number];
}

export class ConvLSTM2DCell extends LSTMCell {
  /** @nocollapse */
  static className = 'ConvLSTM2DCell';

  readonly filters: number;
  readonly kernelSize: number[];
  readonly strides: number[];
  readonly padding: PaddingMode;
  readonly dataFormat: DataFormat;
  readonly dilationRate: number[];

  constructor(args: ConvLSTM2DCellArgs) {
    const {
      filters,
      kernelSize,
      strides,
      padding,
      dataFormat,
      dilationRate,
    } = args;

    super({...args, units: filters});

    this.filters = filters;
    assertPositiveInteger(this.filters, 'filters');

    this.kernelSize = normalizeArray(kernelSize, 2, 'kernelSize');
    this.kernelSize.map(size => assertPositiveInteger(size, 'kernelSize'));

    this.strides = normalizeArray(strides || 1, 2, 'strides');
    this.strides.map(stride => assertPositiveInteger(stride, 'strides'));

    this.padding = padding || 'valid';
    checkPaddingMode(this.padding);

    this.dataFormat = dataFormat || 'channelsLast';
    checkDataFormat(this.dataFormat);

    this.dilationRate = normalizeArray(dilationRate || 1, 2, 'dilationRate');
    this.dilationRate.map(rate => assertPositiveInteger(rate, 'dilationRate'));
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = getExactlyOneShape(inputShape);

    const channelAxis =
        this.dataFormat === 'channelsFirst' ? 1 : inputShape.length - 1;

    if (inputShape[channelAxis] == null) {
      throw new ValueError(
          `The channel dimension of the input should be defined. ` +
          `Found ${inputShape[channelAxis]}`);
    }

    const inputDim = inputShape[channelAxis];

    const kernelShape = this.kernelSize.concat([inputDim, this.filters * 4]);

    this.kernel = this.addWeight(
        'kernel', kernelShape, null, this.kernelInitializer,
        this.kernelRegularizer, true, this.kernelConstraint);

    const recurrentKernelShape =
        this.kernelSize.concat([this.filters, this.filters * 4]);

    this.recurrentKernel = this.addWeight(
        'recurrentKernel', recurrentKernelShape, null,
        this.recurrentInitializer, this.recurrentRegularizer, true,
        this.recurrentConstraint);

    if (this.useBias) {
      let biasInitializer: Initializer;

      if (this.unitForgetBias) {
        const init = this.biasInitializer;

        const filters = this.units;

        biasInitializer = new (class CustomInit extends Initializer {
          /** @nocollapse */
          static className = 'CustomInit';

          apply(shape: Shape, dtype?: DataType): tfc.Tensor {
            const biasI = init.apply([filters]);
            const biasF = tfc.ones([filters]);
            const biasCAndO = init.apply([filters * 2]);
            return K.concatenate([biasI, biasF, biasCAndO]);
          }
        })();
      } else {
        biasInitializer = this.biasInitializer;
      }

      this.bias = this.addWeight(
          'bias', [this.filters * 4], null, biasInitializer,
          this.biasRegularizer, true, this.biasConstraint);
    }

    this.built = true;
  }

  call(inputs: tfc.Tensor[], kwargs: Kwargs): tfc.Tensor[] {
    return tfc.tidy(() => {
      if (inputs.length !== 3) {
        throw new ValueError(
            `ConvLSTM2DCell expects 3 input Tensors (inputs, h, c), got ` +
            `${inputs.length}.`);
      }

      const training = kwargs['training'] || false;

      const x = inputs[0];         // Current input
      const hTMinus1 = inputs[1];  // Previous memory state.
      const cTMinus1 = inputs[2];  // Previous carry state.

      const toCreateDropout =
          (dropout: number, existingMask: tfc.Tensor|tfc.Tensor[]) =>
              0 < dropout && dropout < 1 && existingMask == null;

      if (toCreateDropout(this.dropout, this.dropoutMask)) {
        this.dropoutMask = this.generateDropoutMask({
          ones: () => tfc.onesLike(x),
          rate: this.dropout,
          training,
          count: 4
        }) as tfc.Tensor[];
      }

      type DropoutMasks = [tfc.Tensor, tfc.Tensor, tfc.Tensor, tfc.Tensor];

      const dropoutMask = this.dropoutMask as DropoutMasks;

      let xI = dropoutMask ? tfc.mul(dropoutMask[0], x) : x;
      let xF = dropoutMask ? tfc.mul(dropoutMask[1], x) : x;
      let xC = dropoutMask ? tfc.mul(dropoutMask[2], x) : x;
      let xO = dropoutMask ? tfc.mul(dropoutMask[3], x) : x;

      if (toCreateDropout(this.recurrentDropout, this.recurrentDropoutMask)) {
        this.recurrentDropoutMask = this.generateDropoutMask({
          ones: () => tfc.onesLike(hTMinus1),
          rate: this.recurrentDropout,
          training,
          count: 4
        }) as tfc.Tensor[];
      }

      const recDropoutMask = this.recurrentDropoutMask as DropoutMasks;

      let hI = recDropoutMask ? tfc.mul(recDropoutMask[0], hTMinus1) : hTMinus1;
      let hF = recDropoutMask ? tfc.mul(recDropoutMask[1], hTMinus1) : hTMinus1;
      let hC = recDropoutMask ? tfc.mul(recDropoutMask[2], hTMinus1) : hTMinus1;
      let hO = recDropoutMask ? tfc.mul(recDropoutMask[3], hTMinus1) : hTMinus1;

      const [kernelI, kernelF, kernelC, kernelO]: tfc.Tensor[] =
          tfc.split(this.kernel.read(), 4, 3);

      const [biasI, biasF, biasC, biasO]: tfc.Tensor[] = this.useBias ?
          tfc.split(this.bias.read(), 4) :
          [null, null, null, null];

      xI = this.inputConv(xI, kernelI, biasI, this.padding);
      xF = this.inputConv(xF, kernelF, biasF, this.padding);
      xC = this.inputConv(xC, kernelC, biasC, this.padding);
      xO = this.inputConv(xO, kernelO, biasO, this.padding);

      const [recKernelI, recKernelF, recKernelC, recKernelO]: tfc.Tensor[] =
          tfc.split(this.recurrentKernel.read(), 4, 3);

      hI = this.recurrentConv(hI, recKernelI);
      hF = this.recurrentConv(hF, recKernelF);
      hC = this.recurrentConv(hC, recKernelC);
      hO = this.recurrentConv(hO, recKernelO);

      const i = this.recurrentActivation.apply(tfc.add(xI, hI));
      const f = this.recurrentActivation.apply(tfc.add(xF, hF));
      const c = tfc.add(
          tfc.mul(f, cTMinus1),
          tfc.mul(i, this.activation.apply(tfc.add(xC, hC))));
      const h = tfc.mul(
          this.recurrentActivation.apply(tfc.add(xO, hO)),
          this.activation.apply(c));

      return [h, h, c];
    });
  }

  getConfig(): tfc.serialization.ConfigDict {
    const config: tfc.serialization.ConfigDict = {
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

    return {...baseConfig, ...config};
  }

  computeOutputShape(inputShape: Shape): Shape|Shape[] {
    const isChannelsFirst = this.dataFormat === 'channelsFirst';

    const h = inputShape[isChannelsFirst ? 3 : 2];
    const w = inputShape[isChannelsFirst ? 4 : 3];

    const hOut = convOutputLength(
        h, this.kernelSize[0], this.padding, this.strides[0],
        this.dilationRate[0]);
    const wOut = convOutputLength(
        w, this.kernelSize[1], this.padding, this.strides[1],
        this.dilationRate[1]);

    const outShape = [
      ...inputShape.slice(0, 2),
      ...(isChannelsFirst ? [this.filters, hOut, wOut] :
                            [hOut, wOut, this.filters])
    ];

    return outShape;
  }

  getInitialState(inputs: tfc.Tensor): tfc.Tensor[] {
    return tfc.tidy(() => {
      let initialState = tfc.zerosLike(inputs);

      initialState = tfc.sum(initialState, 1);

      const shape = [...this.kernel.shape.slice(0, -1), this.filters];

      initialState =
          this.inputConv(initialState, tfc.zeros(shape), null, this.padding);

      if (Array.isArray(this.stateSize)) {
        return Array(this.stateSize.length).fill(initialState);
      }

      return [initialState];
    });
  }

  protected generateDropoutMask(args: {
    ones: () => tfc.Tensor,
    rate: number,
    training?: boolean,
    count?: number,
  }): tfc.Tensor|tfc.Tensor[] {
    const {ones, rate, training = false, count = 1} = args;

    const droppedInputs = () => K.dropout(ones(), rate);

    const createMask = () => K.inTrainPhase(droppedInputs, ones, training);

    if (count === 1) {
      return tfc.keep(createMask().clone());
    }

    const masks = Array(count).map(createMask);

    return masks.map(m => tfc.keep(m.clone()));
  }

  protected inputConv(
      x: tfc.Tensor,
      w: tfc.Tensor,
      b: tfc.Tensor = null,
      padding: PaddingMode = 'valid',
  ) {
    const out = tfc.conv2d(
        x as tfc.Tensor3D, w as tfc.Tensor4D, this.strides as [number, number],
        padding as 'same' | 'valid',
        this.dataFormat === 'channelsFirst' ? 'NCHW' : 'NHWC',
        this.dilationRate as [number, number]);

    if (b) {
      return K.biasAdd(out, b, this.dataFormat) as tfc.Tensor3D;
    }

    return out;
  }

  protected recurrentConv(x: tfc.Tensor, w: tfc.Tensor) {
    return tfc.conv2d(
        x as tfc.Tensor3D, w as tfc.Tensor4D, 1, 'same',
        this.dataFormat === 'channelsFirst' ? 'NCHW' : 'NHWC');
  }
}

tfc.serialization.registerClass(ConvLSTM2DCell);

export declare interface ConvLSTM2DArgs extends Omit<LSTMLayerArgs, 'units'>,
                                                ConvLSTM2DCellArgs {}

export class ConvLSTM2D extends RNN {
  /** @nocollapse */
  static className = 'ConvLSTM2D';

  constructor(args: ConvLSTM2DArgs) {
    if (args.implementation === 0) {
      console.warn(
          '`implementation=0` has been deprecated, and now defaults to ' +
          '`implementation=1`. Please update your layer call.');
    }

    args.cell = new ConvLSTM2DCell(args);

    super(args as RNNLayerArgs);

    this.inputSpec = [new InputSpec({ndim: 5})];
  }

  call(inputs: tfc.Tensor|tfc.Tensor[], kwargs: Kwargs): tfc.Tensor
      |tfc.Tensor[] {
    return tfc.tidy(() => {
      if (this.cell.dropoutMask != null) {
        tfc.dispose(this.cell.dropoutMask);
        this.cell.dropoutMask = null;
      }

      if (this.cell.recurrentDropoutMask != null) {
        tfc.dispose(this.cell.recurrentDropoutMask);
        this.cell.recurrentDropoutMask = null;
      }

      const mask = kwargs == null ? null : kwargs['mask'];

      const training = kwargs == null ? null : kwargs['training'];

      const initialState: tfc.Tensor[] =
          kwargs == null ? null : kwargs['initialState'];

      return super.call(inputs, {mask, training, initialState});
    });
  }

  computeOutputShape(inputShape: Shape): Shape|Shape[] {
    let outShape = (this.cell as ConvLSTM2DCell).computeOutputShape(inputShape);

    if (!this.returnSequences) {
      outShape =
          [...outShape.slice(0, 1) as Shape[], ...outShape.slice(2) as Shape[]];
    }

    if (this.returnState) {
      outShape =
          [outShape, ...Array(2).fill([inputShape[0], ...outShape.slice(-3)])];
    }

    return outShape;
  }

  getInitialState(inputs: tfc.Tensor): tfc.Tensor[] {
    return tfc.tidy(() => {
      return (this.cell as ConvLSTM2DCell).getInitialState(inputs);
    });
  }

  getConfig(): tfc.serialization.ConfigDict {
    const {
      filters,
      kernelSize,
      strides,
      padding,
      dataFormat,
      dilationRate,
      activation,
      recurrentActivation,
      useBias,
      kernelInitializer,
      recurrentInitializer,
      biasInitializer,
      unitForgetBias,
      kernelRegularizer,
      recurrentRegularizer,
      biasRegularizer,
      activityRegularizer,
      kernelConstraint,
      recurrentConstraint,
      biasConstraint,
      dropout,
      recurrentDropout,
      implementation,
    } = this.cell as ConvLSTM2DCell;

    const config: tfc.serialization.ConfigDict = {
      filters,
      kernelSize,
      strides,
      padding,
      dataFormat,
      dilationRate,
      activation: serializeActivation(activation),
      recurrentActivation: serializeActivation(recurrentActivation),
      useBias,
      kernelInitializer: serializeInitializer(kernelInitializer),
      recurrentInitializer: serializeInitializer(recurrentInitializer),
      biasInitializer: serializeInitializer(biasInitializer),
      unitForgetBias,
      kernelRegularizer: serializeRegularizer(kernelRegularizer),
      recurrentRegularizer: serializeRegularizer(recurrentRegularizer),
      biasRegularizer: serializeRegularizer(biasRegularizer),
      activityRegularizer: serializeRegularizer(activityRegularizer),
      kernelConstraint: serializeConstraint(kernelConstraint),
      recurrentConstraint: serializeConstraint(recurrentConstraint),
      biasConstraint: serializeConstraint(biasConstraint),
      dropout,
      recurrentDropout,
      implementation,
    };

    const {'cell': _, ...baseConfig} = super.getConfig();

    return {...baseConfig, ...config};
  }

  /** @nocollapse */
  static fromConfig<T extends tfc.serialization.Serializable>(
      cls: tfc.serialization.SerializableConstructor<T>,
      config: tfc.serialization.ConfigDict): T {
    if (config['implmentation'] === 0) {
      config['implementation'] = 1;
    }

    return new cls(config);
  }
}

tfc.serialization.registerClass(ConvLSTM2D);
