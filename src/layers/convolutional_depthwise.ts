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

// tslint:disable:max-line-length
import {serialization, Tensor} from '@tensorflow/tfjs-core';

// tslint:disable:max-line-length
import * as K from '../backend/tfjs_backend';
import {Constraint, ConstraintIdentifier, getConstraint} from '../constraints';
import {ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier} from '../initializers';
import {getRegularizer, Regularizer, RegularizerIdentifier} from '../regularizers';
import {LayerVariable, Shape} from '../types';
import {convOutputLength} from '../utils/conv_utils';
import {getExactlyOneShape, getExactlyOneTensor} from '../utils/generic_utils';

import {Conv2D, ConvLayerConfig} from './convolutional';
// tslint:enable:max-line-length

export interface DepthwiseConv2DLayerConfig extends ConvLayerConfig {
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

/**
 * Depthwise separable 2D convolution.
 *
 * Depthwise Separable convolutions consists in performing just the first step
 * in a depthwise spatial convolution (which acts on each input channel
 * separately). The `depthMultplier` argument controls how many output channels
 * are generated per input channel in the depthwise step.
 */
export class DepthwiseConv2D extends Conv2D {
  static className = 'DepthwiseConv2D';
  private readonly depthMultiplier: number;
  private readonly depthwiseInitializer: Initializer;
  private readonly depthwiseConstraint: Constraint;
  private readonly depthwiseRegularizer: Regularizer;

  private depthwiseKernel: LayerVariable = null;

  constructor(config: DepthwiseConv2DLayerConfig) {
    super(config);
    this.depthMultiplier =
        config.depthMultiplier == null ? 1 : config.depthMultiplier;
    this.depthwiseInitializer = getInitializer(
        config.depthwiseInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.depthwiseConstraint = getConstraint(config.depthwiseConstraint);
    this.depthwiseRegularizer = getRegularizer(config.depthwiseRegularizer);
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

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    inputs = getExactlyOneTensor(inputs);
    let outputs = K.depthwiseConv2d(
        inputs, this.depthwiseKernel.read(), this.strides as [number, number],
        this.padding, this.dataFormat, null);
    // TODO(cais): Add support for dilation.
    if (this.useBias) {
      outputs = K.biasAdd(outputs, this.bias.read(), this.dataFormat);
    }
    if (this.activation != null) {
      outputs = this.activation(outputs);
    }
    return outputs;
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
}
serialization.SerializationMap.register(DepthwiseConv2D);
