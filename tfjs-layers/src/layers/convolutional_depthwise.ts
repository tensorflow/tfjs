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
 * TensorFlow.js Layers: Depthwise Convolutional Layers
 */

import * as tfc from '@tensorflow/tfjs-core';
import {serialization, Tensor, Tensor4D, tidy} from '@tensorflow/tfjs-core';

import {imageDataFormat} from '../backend/common';
import * as K from '../backend/tfjs_backend';
import {checkDataFormat} from '../common';
import {Constraint, ConstraintIdentifier, getConstraint, serializeConstraint} from '../constraints';
import {ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, serializeInitializer} from '../initializers';
import {DataFormat, Shape} from '../keras_format/common';
import {getRegularizer, Regularizer, RegularizerIdentifier, serializeRegularizer} from '../regularizers';
import {Kwargs} from '../types';
import {convOutputLength} from '../utils/conv_utils';
import {getExactlyOneShape, getExactlyOneTensor} from '../utils/types_utils';
import {LayerVariable} from '../variables';

import {BaseConv, BaseConvLayerArgs, ConvLayerArgs, preprocessConv2DInput} from './convolutional';

/**
 * 2D convolution with separable filters.
 * @param x Input tensor.
 * @param depthwiseKernel Convolution kernel for depthwise convolution.
 * @param strides Strides (Array of two integers).
 * @param padding Padding model.
 * @param dataFormat Data format.
 * @param dilationRate Array of two integers, dilation rates for the separable
 *   convolution.
 * @returns Output tensor.
 * @throws ValueError If depthwiseKernel is not a 4D array.
 */
export function depthwiseConv2d(
    x: Tensor, depthwiseKernel: Tensor, strides: [number, number] = [1, 1],
    padding = 'valid', dataFormat?: DataFormat,
    dilationRate?: [number, number]): Tensor {
  return tidy(() => {
    if (dataFormat == null) {
      dataFormat = imageDataFormat();
    }
    checkDataFormat(dataFormat);
    let y = preprocessConv2DInput(x, dataFormat);
    if (x.rank !== 4) {
      throw new ValueError(
          `Input for depthwiseConv2d is required to be 4-D, but is instead ` +
          `${x.rank}-D`);
    }
    if (depthwiseKernel.rank !== 4) {
      throw new ValueError(
          `depthwiseKernel is required to be 4-D, but is instead ` +
          `${depthwiseKernel.rank}-D`);
    }
    y = tfc.depthwiseConv2d(
        y as Tensor4D, depthwiseKernel as Tensor4D, strides,
        padding === 'same' ? 'same' : 'valid', 'NHWC', dilationRate);
    if (dataFormat === 'channelsFirst') {
      y = tfc.transpose(y, [0, 3, 1, 2]);
    }
    return y;
  });
}

export declare interface DepthwiseConv2DLayerArgs extends BaseConvLayerArgs {
  /**
   * An integer or Array of 2 integers, specifying the width and height of the
   * 2D convolution window. Can be a single integer to specify the same value
   * for all spatial dimensions.
   */
  kernelSize: number|[number, number];

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
   * Default: GlorotNormal.
   */
  depthwiseInitializer?: InitializerIdentifier|Initializer;

  /**
   * Constraint for the depthwise kernel matrix.
   */
  depthwiseConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Regulzarizer function for the depthwise kernel matrix.
   */
  depthwiseRegularizer?: RegularizerIdentifier|Regularizer;
}

export class DepthwiseConv2D extends BaseConv {
  /** @nocollapse */
  static className = 'DepthwiseConv2D';
  private readonly depthMultiplier: number;
  private readonly depthwiseInitializer: Initializer;
  private readonly depthwiseConstraint: Constraint;
  private readonly depthwiseRegularizer: Regularizer;

  private depthwiseKernel: LayerVariable = null;

  constructor(args: DepthwiseConv2DLayerArgs) {
    super(2, args as ConvLayerArgs);
    this.depthMultiplier =
        args.depthMultiplier == null ? 1 : args.depthMultiplier;
    this.depthwiseInitializer = getInitializer(
        args.depthwiseInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.depthwiseConstraint = getConstraint(args.depthwiseConstraint);
    this.depthwiseRegularizer = getRegularizer(args.depthwiseRegularizer);
  }

  build(inputShape: Shape|Shape[]): void {
    inputShape = getExactlyOneShape(inputShape);
    if (inputShape.length < 4) {
      throw new ValueError(
          `Inputs to DepthwiseConv2D should have rank 4. ` +
          `Received input shape: ${JSON.stringify(inputShape)}.`);
    }
    const channelAxis = this.dataFormat === 'channelsFirst' ? 1 : 3;
    if (inputShape[channelAxis] == null || inputShape[channelAxis] < 0) {
      throw new ValueError(
          'The channel dimension of the inputs to DepthwiseConv2D should ' +
          `be defined, but is not (${inputShape[channelAxis]}).`);
    }
    const inputDim = inputShape[channelAxis];
    const depthwiseKernelShape: Shape = [
      this.kernelSize[0], this.kernelSize[1], inputDim, this.depthMultiplier
    ];

    this.depthwiseKernel = this.addWeight(
        'depthwise_kernel', depthwiseKernelShape, null,
        this.depthwiseInitializer, this.depthwiseRegularizer, true,
        this.depthwiseConstraint);
    if (this.useBias) {
      this.bias = this.addWeight(
          'bias', [inputDim * this.depthMultiplier], null, this.biasInitializer,
          this.biasRegularizer, true, this.biasConstraint);
    } else {
      this.bias = null;
    }
    this.built = true;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      inputs = getExactlyOneTensor(inputs);
      let outputs = depthwiseConv2d(
          inputs, this.depthwiseKernel.read(), this.strides as [number, number],
          this.padding, this.dataFormat, null);
      // TODO(cais): Add support for dilation.
      if (this.useBias) {
        outputs = K.biasAdd(outputs, this.bias.read(), this.dataFormat);
      }
      if (this.activation != null) {
        outputs = this.activation.apply(outputs);
      }
      return outputs;
    });
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    const rows =
        this.dataFormat === 'channelsFirst' ? inputShape[2] : inputShape[1];
    const cols =
        this.dataFormat === 'channelsFirst' ? inputShape[3] : inputShape[2];
    const outFilters = this.dataFormat === 'channelsFirst' ?
        inputShape[1] * this.depthMultiplier :
        inputShape[3] * this.depthMultiplier;
    const outRows = convOutputLength(
        rows, this.kernelSize[0], this.padding, this.strides[0]);
    const outCols = convOutputLength(
        cols, this.kernelSize[1], this.padding, this.strides[1]);
    if (this.dataFormat === 'channelsFirst') {
      return [inputShape[0], outFilters, outRows, outCols];
    } else {
      // In this case, assume 'channelsLast'.
      return [inputShape[0], outRows, outCols, outFilters];
    }
  }

  getConfig(): serialization.ConfigDict {
    const config = super.getConfig();
    config['depthMultiplier'] = this.depthMultiplier;
    config['depthwiseInitializer'] =
        serializeInitializer(this.depthwiseInitializer);
    config['depthwiseRegularizer'] =
        serializeRegularizer(this.depthwiseRegularizer);
    config['depthwiseConstraint'] =
        serializeConstraint(this.depthwiseRegularizer);
    return config;
  }
}
serialization.registerClass(DepthwiseConv2D);
