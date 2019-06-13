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
 * TensorFlow.js Layers: Noise Layers.
 */


import {greaterEqual, randomUniform, serialization, Tensor, tidy} from '@tensorflow/tfjs-core';

import * as K from '../backend/tfjs_backend';
import {Layer, LayerArgs} from '../engine/topology';
import {Shape} from '../keras_format/common';
import {Kwargs} from '../types';
import {getExactlyOneTensor} from '../utils/types_utils';

export declare interface GaussianNoiseArgs extends LayerArgs {
  /** Standard Deviation.  */
  stddev: number;
}


export class GaussianNoise extends Layer {
  static className = 'GaussianNoise';
  readonly stddev: number;

  constructor(args: GaussianNoiseArgs) {
    super(args);
    this.supportsMasking = true;
    this.stddev = args.stddev;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig() {
    const baseConfig = super.getConfig();
    const config = {stddev: this.stddev};
    Object.assign(config, baseConfig);
    return config;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      const input = getExactlyOneTensor(inputs);
      const noised = () =>
          K.randomNormal(input.shape, 0, this.stddev).add(input);
      const output =
          K.inTrainPhase(noised, () => input, kwargs['training'] || false) as
          Tensor;
      return output;
    });
  }
}
serialization.registerClass(GaussianNoise);

export declare interface GaussianDropoutArgs extends LayerArgs {
  /** drop probability.  */
  rate: number;
}

export class GaussianDropout extends Layer {
  static className = 'GaussianDropout';
  readonly rate: number;

  constructor(args: GaussianDropoutArgs) {
    super(args);
    this.supportsMasking = true;
    this.rate = args.rate;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig() {
    const baseConfig = super.getConfig();
    const config = {rate: this.rate};
    Object.assign(config, baseConfig);
    return config;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      const input = getExactlyOneTensor(inputs);
      if (this.rate > 0 && this.rate < 1) {
        const noised = () => {
          const stddev = Math.sqrt(this.rate / (1 - this.rate));
          return input.mul(K.randomNormal(input.shape, 1, stddev));
        };
        return K.inTrainPhase(noised, () => input, kwargs['training'] || false);
      }
      return input;
    });
  }
}
serialization.registerClass(GaussianDropout);

export declare interface AlphaDropoutArgs extends LayerArgs {
  /** drop probability.  */
  rate: number;
  /**
   * A 1-D `Tensor` of type `int32`, representing the
   * shape for randomly generated keep/drop flags.
   */
  noiseShape?: Shape;
}

/**
 * Applies Alpha Dropout to the input.
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
 * to their original values, in order to ensure the self-normalizing property
 * even after this dropout.
 * Alpha Dropout fits well to Scaled Exponential Linear Units
 * by randomly setting activations to the negative saturation value.
 *
 * Arguments:
 *   - `rate`: float, drop probability (as with `Dropout`).
 *     The multiplicative noise will have
 *     standard deviation `sqrt(rate / (1 - rate))`.
 *   - `noise_shape`: A 1-D `Tensor` of type `int32`, representing the
 *     shape for randomly generated keep/drop flags.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape`
 *   (tuple of integers, does not include the samples axis)
 *   when using this layer as the first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
 */
export class AlphaDropout extends Layer {
  static className = 'AlphaDropout';
  readonly rate: number;
  readonly noiseShape: Shape;

  constructor(args: AlphaDropoutArgs) {
    super(args);
    this.supportsMasking = true;
    this.rate = args.rate;
    this.noiseShape = args.noiseShape;
  }

  _getNoiseShape(inputs: Tensor|Tensor[]) {
    return this.noiseShape || getExactlyOneTensor(inputs).shape;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig() {
    const baseConfig = super.getConfig();
    const config = {rate: this.rate};
    Object.assign(config, baseConfig);
    return config;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      if (this.rate < 1 && this.rate > 0) {
        const noiseShape = this._getNoiseShape(inputs);

        const droppedInputs = () => {
          const input = getExactlyOneTensor(inputs);

          const alpha = 1.6732632423543772848170429916717;
          const scale = 1.0507009873554804934193349852946;

          const alphaP = -alpha * scale;

          let keptIdx = greaterEqual(randomUniform(noiseShape), this.rate);

          keptIdx = K.cast(keptIdx, 'float32');  // get default dtype.

          // Get affine transformation params.
          const a = ((1 - this.rate) * (1 + this.rate * alphaP ** 2)) ** -0.5;
          const b = -a * alphaP * this.rate;

          // Apply mask.
          const x = input.mul(keptIdx).add(keptIdx.add(-1).mul(alphaP));

          return x.mul(a).add(b);
        };
        return K.inTrainPhase(
            droppedInputs, () => getExactlyOneTensor(inputs),
            kwargs['training'] || false);
      }
      return inputs;
    });
  }
}
serialization.registerClass(AlphaDropout);
