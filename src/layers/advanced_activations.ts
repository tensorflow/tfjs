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
 *  Advanced activation layers.
 */

import {clipByValue, elu, leakyRelu, prelu, relu, serialization, Tensor} from '@tensorflow/tfjs-core';

import {Softmax as softmaxActivation} from '../activations';
import {cast} from '../backend/tfjs_backend';
import {Constraint, getConstraint, serializeConstraint} from '../constraints';
import {InputSpec, Layer, LayerArgs} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, serializeInitializer} from '../initializers';
import {Shape} from '../keras_format/common';
import {getRegularizer, Regularizer, serializeRegularizer} from '../regularizers';
import {Kwargs} from '../types';
import {getExactlyOneShape, getExactlyOneTensor} from '../utils/types_utils';
import {LayerVariable} from '../variables';

export declare interface ReLULayerArgs extends LayerArgs {
  /**
   * Float, the maximum output value.
   */
  maxValue?: number;
}

/**
 * Rectified Linear Unit activation function.
 *
 * Input shape:
 *   Arbitrary. Use the config field `inputShape` (Array of integers, does
 *   not include the sample axis) when using this layer as the first layer
 *   in a model.
 *
 * Output shape:
 *   Same shape as the input.
 */
export class ReLU extends Layer {
  /** @nocollapse */
  static className = 'ReLU';
  maxValue: number;

  constructor(args?: ReLULayerArgs) {
    super(args == null ? {} : args);
    this.supportsMasking = true;
    if (args != null) {
      this.maxValue = args.maxValue;
    }
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    inputs = getExactlyOneTensor(inputs);
    let output = relu(inputs);
    if (this.maxValue != null) {
      output = clipByValue(output, 0, this.maxValue);
    }
    return output;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {maxValue: this.maxValue};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(ReLU);

export declare interface LeakyReLULayerArgs extends LayerArgs {
  /**
   * Float `>= 0`. Negative slope coefficient. Defaults to `0.3`.
   */
  alpha?: number;
}

/**
 * Leaky version of a rectified linear unit.
 *
 * It allows a small gradient when the unit is not active:
 * `f(x) = alpha * x for x < 0.`
 * `f(x) = x for x >= 0.`
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 */
export class LeakyReLU extends Layer {
  /** @nocollapse */
  static className = 'LeakyReLU';
  readonly alpha: number;

  readonly DEFAULT_ALPHA = 0.3;

  constructor(args?: LeakyReLULayerArgs) {
    super(args == null ? {} : args);
    if (args == null) {
      args = {};
    }
    this.alpha = args.alpha == null ? this.DEFAULT_ALPHA : args.alpha;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const x = getExactlyOneTensor(inputs);
    return leakyRelu(x, this.alpha);
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {alpha: this.alpha};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(LeakyReLU);

export declare interface PReLULayerArgs extends LayerArgs {
  /**
   * Initializer for the learnable alpha.
   */
  alphaInitializer?: Initializer|InitializerIdentifier;

  /**
   * Regularizer for the learnable alpha.
   */
  alphaRegularizer?: Regularizer;

  /**
   * Constraint for the learnable alpha.
   */
  alphaConstraint?: Constraint;

  /**
   * The axes along which to share learnable parameters for the activation
   * function. For example, if the incoming feature maps are from a 2D
   * convolution with output shape `[numExamples, height, width, channels]`,
   * and you wish to share parameters across space (height and width) so that
   * each filter channels has only one set of parameters, set
   * `shared_axes: [1, 2]`.
   */
  sharedAxes?: number|number[];
}

/**
 * Parameterized version of a leaky rectified linear unit.
 *
 * It follows
 * `f(x) = alpha * x for x < 0.`
 * `f(x) = x for x >= 0.`
 * wherein `alpha` is a trainable weight.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 */
export class PReLU extends Layer {
  /** @nocollapse */
  static className = 'PReLU';
  private readonly alphaInitializer: Initializer;
  private readonly alphaRegularizer: Regularizer;
  private readonly alphaConstraint: Constraint;
  private readonly sharedAxes: number[];
  private alpha: LayerVariable;

  readonly DEFAULT_ALPHA_INITIALIZER: InitializerIdentifier = 'zeros';

  constructor(args?: PReLULayerArgs) {
    super(args == null ? {} : args);
    if (args == null) {
      args = {};
    }

    this.supportsMasking = true;
    this.alphaInitializer =
        getInitializer(args.alphaInitializer || this.DEFAULT_ALPHA_INITIALIZER);
    this.alphaRegularizer = getRegularizer(args.alphaRegularizer);
    this.alphaConstraint = getConstraint(args.alphaConstraint);
    if (args.sharedAxes == null) {
      this.sharedAxes = null;
    } else if (Array.isArray(args.sharedAxes)) {
      this.sharedAxes = args.sharedAxes;
    } else if (typeof args.sharedAxes === 'number') {
      this.sharedAxes = [args.sharedAxes];
    } else {
      throw new ValueError(
          `Expected sharedAxes to be a number or an array of numbers, ` +
          `but got ${args.sharedAxes}`);
    }
  }

  build(inputShape: Shape|Shape[]) {
    inputShape = getExactlyOneShape(inputShape);
    const paramShape: Shape = inputShape.slice(1);
    if (this.sharedAxes != null) {
      for (const i of this.sharedAxes) {
        paramShape[i - 1] = 1;
      }
    }
    this.alpha = this.addWeight(
        'alpha', paramShape, 'float32', this.alphaInitializer,
        this.alphaRegularizer, true, this.alphaConstraint);
    // Set input spec.
    const axes: {[axis: number]: number} = {};
    if (this.sharedAxes != null) {
      for (let i = 1; i < inputShape.length; ++i) {
        axes[i] = inputShape[i];
      }
    }
    this.inputSpec = [new InputSpec({
      ndim: inputShape.length,
      axes,
    })];
    this.built = true;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    inputs = getExactlyOneTensor(inputs);
    return prelu(inputs, this.alpha.read());
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      alphaInitializer: serializeInitializer(this.alphaInitializer),
      alphaRegularizer: serializeRegularizer(this.alphaRegularizer),
      alphaConstraint: serializeConstraint(this.alphaConstraint),
      sharedAxes: this.sharedAxes
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(PReLU);

export declare interface ELULayerArgs extends LayerArgs {
  /**
   * Float `>= 0`. Negative slope coefficient. Defaults to `1.0`.
   */
  alpha?: number;
}

/**
 * Exponetial Linear Unit (ELU).
 *
 * It follows:
 * `f(x) =  alpha * (exp(x) - 1.) for x < 0`,
 * `f(x) = x for x >= 0`.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * References:
 *   - [Fast and Accurate Deep Network Learning by Exponential Linear Units
 * (ELUs)](https://arxiv.org/abs/1511.07289v1)
 */
export class ELU extends Layer {
  /** @nocollapse */
  static className = 'ELU';
  readonly alpha: number;

  readonly DEFAULT_ALPHA = 1.0;

  constructor(args?: ELULayerArgs) {
    super(args == null ? {} : args);
    if (args == null) {
      args = {};
    }

    if (args.alpha != null && args.alpha !== this.DEFAULT_ALPHA) {
      throw new NotImplementedError(
          `Non-default alpha value (${args.alpha}) is not supported by the ` +
          `ELU layer yet.`);
    }

    this.alpha = args.alpha == null ? this.DEFAULT_ALPHA : args.alpha;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const x = getExactlyOneTensor(inputs);
    return elu(x);
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {alpha: this.alpha};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(ELU);

export declare interface ThresholdedReLULayerArgs extends LayerArgs {
  /**
   * Float >= 0. Threshold location of activation.
   */
  theta?: number;
}

/**
 * Thresholded Rectified Linear Unit.
 *
 * It follows:
 * `f(x) = x for x > theta`,
 * `f(x) = 0 otherwise`.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * References:
 *   - [Zero-Bias Autoencoders and the Benefits of Co-Adapting
 * Features](http://arxiv.org/abs/1402.3337)
 */
export class ThresholdedReLU extends Layer {
  /** @nocollapse */
  static className = 'ThresholdedReLU';
  readonly theta: number;

  readonly DEFAULT_THETA = 1.0;

  constructor(args?: ThresholdedReLULayerArgs) {
    super(args == null ? {} : args);
    if (args == null) {
      args = {};
    }

    this.theta = args.theta == null ? this.DEFAULT_THETA : args.theta;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const x = getExactlyOneTensor(inputs);
    return x.mul(cast(x.greater(this.theta), 'float32'));
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {theta: this.theta};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(ThresholdedReLU);

export declare interface SoftmaxLayerArgs extends LayerArgs {
  /**
   * Integer, axis along which the softmax normalization is applied.
   * Defaults to `-1` (i.e., the last axis).
   */
  axis?: number;
}

/**
 * Softmax activation layer.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 */
export class Softmax extends Layer {
  /** @nocollapse */
  static className = 'Softmax';
  readonly axis: number;
  readonly softmax: (t: Tensor, a?: number) => Tensor;
  readonly DEFAULT_AXIS = 1.0;

  constructor(args?: SoftmaxLayerArgs) {
    super(args == null ? {} : args);
    if (args == null) {
      args = {};
    }
    this.softmax = new softmaxActivation().apply;
    this.axis = args.axis == null ? this.DEFAULT_AXIS : args.axis;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const x = getExactlyOneTensor(inputs);
    return this.softmax(x, this.axis);
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {axis: this.axis};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(Softmax);
