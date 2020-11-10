/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';
import {Tensor, util} from '@tensorflow/tfjs-core';

import {Activation} from '../activations';
import * as K from '../backend/tfjs_backend';
import {checkDataFormat, checkPaddingMode} from '../common';
import {Constraint} from '../constraints';
import {InputSpec} from '../engine/topology';
import {AttributeError, NotImplementedError, ValueError} from '../errors';
import {Initializer} from '../initializers';
import {DataFormat, DataType, PaddingMode, Shape} from '../keras_format/common';
import {Regularizer} from '../regularizers';
import {Kwargs} from '../types';
import {convOutputLength, normalizeArray} from '../utils/conv_utils';
import {assertPositiveInteger} from '../utils/generic_utils';
import {getExactlyOneShape} from '../utils/types_utils';

import {BaseRNNLayerArgs, generateDropoutMask, LSTMCell, LSTMCellLayerArgs, LSTMLayerArgs, RNN, RNNCell, RNNLayerArgs, SimpleRNNCellLayerArgs} from './recurrent';

declare interface ConvRNN2DCellArgs extends
    Omit<SimpleRNNCellLayerArgs, 'units'> {
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

abstract class ConvRNN2DCell extends RNNCell {
  readonly filters: number;
  readonly kernelSize: number[];
  readonly strides: number[];
  readonly padding: PaddingMode;
  readonly dataFormat: DataFormat;
  readonly dilationRate: number[];

  readonly activation: Activation;
  readonly useBias: boolean;

  readonly kernelInitializer: Initializer;
  readonly recurrentInitializer: Initializer;
  readonly biasInitializer: Initializer;

  readonly kernelConstraint: Constraint;
  readonly recurrentConstraint: Constraint;
  readonly biasConstraint: Constraint;

  readonly kernelRegularizer: Regularizer;
  readonly recurrentRegularizer: Regularizer;
  readonly biasRegularizer: Regularizer;

  readonly dropout: number;
  readonly recurrentDropout: number;
}

declare interface ConvRNN2DLayerArgs extends BaseRNNLayerArgs,
                                             ConvRNN2DCellArgs {}

/**
 * Base class for convolutional-recurrent layers.
 */
class ConvRNN2D extends RNN {
  /** @nocollapse */
  static className = 'ConvRNN2D';

  readonly cell: ConvRNN2DCell;

  constructor(args: ConvRNN2DLayerArgs) {
    if (args.unroll) {
      throw new NotImplementedError(
          'Unrolling is not possible with convolutional RNNs.');
    }

    if (Array.isArray(args.cell)) {
      throw new NotImplementedError(
          'It is not possible at the moment to stack convolutional cells.');
    }

    super(args as RNNLayerArgs);

    this.inputSpec = [new InputSpec({ndim: 5})];
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tfc.tidy(() => {
      if (this.cell.dropoutMask != null) {
        tfc.dispose(this.cell.dropoutMask);

        this.cell.dropoutMask = null;
      }

      if (this.cell.recurrentDropoutMask != null) {
        tfc.dispose(this.cell.recurrentDropoutMask);

        this.cell.recurrentDropoutMask = null;
      }

      if (kwargs && kwargs['constants']) {
        throw new ValueError('ConvRNN2D cell does not support constants');
      }

      const mask = kwargs == null ? null : kwargs['mask'];

      const training = kwargs == null ? null : kwargs['training'];

      const initialState: Tensor[] =
          kwargs == null ? null : kwargs['initialState'];

      return super.call(inputs, {mask, training, initialState});
    });
  }

  computeOutputShape(inputShape: Shape): Shape|Shape[] {
    let outShape: Shape = this.computeSingleOutputShape(inputShape);

    if (!this.returnSequences) {
      outShape = [outShape[0], ...outShape.slice(2)];
    }

    if (this.returnState) {
      outShape =
          [outShape, ...Array(2).fill([inputShape[0], ...outShape.slice(-3)])];
    }

    return outShape;
  }

  getInitialState(inputs: tfc.Tensor): tfc.Tensor[] {
    return tfc.tidy(() => {
      const {stateSize} = this.cell;

      const inputShape = inputs.shape;

      const outputShape = this.computeSingleOutputShape(inputShape);

      const stateShape = [outputShape[0], ...outputShape.slice(2)];

      const initialState = tfc.zeros(stateShape);

      if (Array.isArray(stateSize)) {
        return Array(stateSize.length).fill(initialState);
      }

      return [initialState];
    });
  }

  resetStates(states?: Tensor|Tensor[], training = false): void {
    tfc.tidy(() => {
      if (!this.stateful) {
        throw new AttributeError(
            'Cannot call resetStates() on an RNN Layer that is not stateful.');
      }

      const inputShape = this.inputSpec[0].shape;

      const outputShape = this.computeSingleOutputShape(inputShape);

      const stateShape = [outputShape[0], ...outputShape.slice(2)];

      const batchSize = inputShape[0];

      if (batchSize == null) {
        throw new ValueError(
            'If an RNN is stateful, it needs to know its batch size. Specify ' +
            'the batch size of your input tensors: \n' +
            '- If using a Sequential model, specify the batch size by ' +
            'passing a `batchInputShape` option to your first layer.\n' +
            '- If using the functional API, specify the batch size by ' +
            'passing a `batchShape` option to your Input layer.');
      }

      // Initialize state if null.
      if (this.getStates() == null) {
        if (Array.isArray(this.cell.stateSize)) {
          this.states_ = this.cell.stateSize.map(() => tfc.zeros(stateShape));
        } else {
          this.states_ = [tfc.zeros(stateShape)];
        }
      } else if (states == null) {
        // Dispose old state tensors.
        tfc.dispose(this.states_);

        // For stateful RNNs, fully dispose kept old states.
        if (this.keptStates != null) {
          tfc.dispose(this.keptStates);
          this.keptStates = [];
        }

        if (Array.isArray(this.cell.stateSize)) {
          this.states_ = this.cell.stateSize.map(() => tfc.zeros(stateShape));
        } else {
          this.states_[0] = tfc.zeros(stateShape);
        }
      } else {
        if (!Array.isArray(states)) {
          states = [states];
        }

        if (states.length !== this.states_.length) {
          throw new ValueError(
              `Layer ${this.name} expects ${this.states_.length} state(s), ` +
              `but it received ${states.length} state value(s). Input ` +
              `received: ${states}`);
        }

        if (training) {
          // Store old state tensors for complete disposal later, i.e., during
          // the next no-arg call to this method. We do not dispose the old
          // states immediately because that BPTT (among other things) require
          // them.
          this.keptStates.push(this.states_.slice());
        } else {
          tfc.dispose(this.states_);
        }

        for (let index = 0; index < this.states_.length; ++index) {
          const value = states[index];

          const expectedShape = stateShape;

          if (!util.arraysEqual(value.shape, expectedShape)) {
            throw new ValueError(
                `State ${index} is incompatible with layer ${this.name}: ` +
                `expected shape=${expectedShape}, received shape=${
                    value.shape}`);
          }

          this.states_[index] = value;
        }
      }

      this.states_ = this.states_.map(state => tfc.keep(state.clone()));
    });
  }

  protected computeSingleOutputShape(inputShape: Shape): Shape {
    const {dataFormat, filters, kernelSize, padding, strides, dilationRate} =
        this.cell;

    const isChannelsFirst = dataFormat === 'channelsFirst';

    const h = inputShape[isChannelsFirst ? 3 : 2];
    const w = inputShape[isChannelsFirst ? 4 : 3];

    const hOut = convOutputLength(
        h, kernelSize[0], padding, strides[0], dilationRate[0]);
    const wOut = convOutputLength(
        w, kernelSize[1], padding, strides[1], dilationRate[1]);

    const outShape: Shape = [
      ...inputShape.slice(0, 2),
      ...(isChannelsFirst ? [filters, hOut, wOut] : [hOut, wOut, filters])
    ];

    return outShape;
  }
}

export declare interface ConvLSTM2DCellArgs extends
    Omit<LSTMCellLayerArgs, 'units'>, ConvRNN2DCellArgs {}

export class ConvLSTM2DCell extends LSTMCell implements ConvRNN2DCell {
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
    this.kernelSize.forEach(size => assertPositiveInteger(size, 'kernelSize'));

    this.strides = normalizeArray(strides || 1, 2, 'strides');
    this.strides.forEach(stride => assertPositiveInteger(stride, 'strides'));

    this.padding = padding || 'valid';
    checkPaddingMode(this.padding);

    this.dataFormat = dataFormat || 'channelsLast';
    checkDataFormat(this.dataFormat);

    this.dilationRate = normalizeArray(dilationRate || 1, 2, 'dilationRate');
    this.dilationRate.forEach(
        rate => assertPositiveInteger(rate, 'dilationRate'));
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

    const numOfKernels = 4;

    const kernelShape =
        this.kernelSize.concat([inputDim, this.filters * numOfKernels]);

    this.kernel = this.addWeight(
        'kernel', kernelShape, null, this.kernelInitializer,
        this.kernelRegularizer, true, this.kernelConstraint);

    const recurrentKernelShape =
        this.kernelSize.concat([this.filters, this.filters * numOfKernels]);

    this.recurrentKernel = this.addWeight(
        'recurrent_kernel', recurrentKernelShape, null,
        this.recurrentInitializer, this.recurrentRegularizer, true,
        this.recurrentConstraint);

    if (this.useBias) {
      let biasInitializer: Initializer;

      if (this.unitForgetBias) {
        const init = this.biasInitializer;

        const filters = this.filters;

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
          'bias', [this.filters * numOfKernels], null, biasInitializer,
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

      const numOfKernels = 4;

      type DropoutMasks = [tfc.Tensor, tfc.Tensor, tfc.Tensor, tfc.Tensor];

      if (0 < this.dropout && this.dropout < 1 && this.dropoutMask == null) {
        this.dropoutMask = generateDropoutMask({
                             ones: () => tfc.onesLike(x),
                             rate: this.dropout,
                             training,
                             count: numOfKernels
                           }) as tfc.Tensor[];
      }

      const dropoutMask = this.dropoutMask as DropoutMasks;

      const applyDropout =
          (x: tfc.Tensor, mask: tfc.Tensor[], index: number) => {
            if (!mask || !mask[index]) {
              return x;
            }

            return tfc.mul(mask[index], x);
          };

      let xI = applyDropout(x, dropoutMask, 0);
      let xF = applyDropout(x, dropoutMask, 1);
      let xC = applyDropout(x, dropoutMask, 2);
      let xO = applyDropout(x, dropoutMask, 3);

      if (0 < this.recurrentDropout && this.recurrentDropout < 1 &&
          this.recurrentDropoutMask == null) {
        this.recurrentDropoutMask = generateDropoutMask({
                                      ones: () => tfc.onesLike(hTMinus1),
                                      rate: this.recurrentDropout,
                                      training,
                                      count: numOfKernels
                                    }) as tfc.Tensor[];
      }

      const recDropoutMask = this.recurrentDropoutMask as DropoutMasks;

      let hI = applyDropout(hTMinus1, recDropoutMask, 0);
      let hF = applyDropout(hTMinus1, recDropoutMask, 1);
      let hC = applyDropout(hTMinus1, recDropoutMask, 2);
      let hO = applyDropout(hTMinus1, recDropoutMask, 3);

      const kernelChannelAxis = 3;

      const [kernelI, kernelF, kernelC, kernelO]: tfc.Tensor[] =
          tfc.split(this.kernel.read(), numOfKernels, kernelChannelAxis);

      const [biasI, biasF, biasC, biasO]: tfc.Tensor[] = this.useBias ?
          tfc.split(this.bias.read(), numOfKernels) :
          [null, null, null, null];

      xI = this.inputConv(xI, kernelI, biasI, this.padding);
      xF = this.inputConv(xF, kernelF, biasF, this.padding);
      xC = this.inputConv(xC, kernelC, biasC, this.padding);
      xO = this.inputConv(xO, kernelO, biasO, this.padding);

      const [recKernelI, recKernelF, recKernelC, recKernelO]: tfc.Tensor[] =
          tfc.split(
              this.recurrentKernel.read(), numOfKernels, kernelChannelAxis);

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
    const {'units': _, ...baseConfig} = super.getConfig();

    const config: tfc.serialization.ConfigDict = {
      filters: this.filters,
      kernelSize: this.kernelSize,
      padding: this.padding,
      dataFormat: this.dataFormat,
      dilationRate: this.dilationRate,
      strides: this.strides,
    };

    return {...baseConfig, ...config};
  }

  inputConv(x: Tensor, w: Tensor, b?: Tensor, padding?: PaddingMode) {
    const out = tfc.conv2d(
        x as tfc.Tensor3D, w as tfc.Tensor4D, this.strides as [number, number],
        (padding || 'valid') as 'same' | 'valid',
        this.dataFormat === 'channelsFirst' ? 'NCHW' : 'NHWC',
        this.dilationRate as [number, number]);

    if (b) {
      return K.biasAdd(out, b, this.dataFormat) as tfc.Tensor3D;
    }

    return out;
  }

  recurrentConv(x: Tensor, w: Tensor) {
    const strides = 1;

    return tfc.conv2d(
        x as tfc.Tensor3D, w as tfc.Tensor4D, strides, 'same',
        this.dataFormat === 'channelsFirst' ? 'NCHW' : 'NHWC');
  }
}

tfc.serialization.registerClass(ConvLSTM2DCell);

export declare interface ConvLSTM2DArgs extends
    Omit<LSTMLayerArgs, 'units'|'cell'>, ConvRNN2DLayerArgs {}

export class ConvLSTM2D extends ConvRNN2D {
  /** @nocollapse */
  static className = 'ConvLSTM2D';

  constructor(args: ConvLSTM2DArgs) {
    const cell = new ConvLSTM2DCell(args);

    super({...args, cell} as ConvRNN2DLayerArgs);
  }

  /** @nocollapse */
  static fromConfig<T extends tfc.serialization.Serializable>(
      cls: tfc.serialization.SerializableConstructor<T>,
      config: tfc.serialization.ConfigDict): T {
    return new cls(config);
  }
}

tfc.serialization.registerClass(ConvLSTM2D);
