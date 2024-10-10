/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { image, Rank, serialization, Tensor, tidy } from '@tensorflow/tfjs-core';
import { getExactlyOneTensor, getExactlyOneShape } from '../../utils/types_utils';
import { Shape } from '../../keras_format/common';
import { Kwargs } from '../../types';
import { ValueError } from '../../errors';
import { BaseRandomLayerArgs, BaseRandomLayer } from '../../engine/base_random_layer';
import { randomUniform } from '@tensorflow/tfjs-core';

export declare interface RandomWidthArgs extends BaseRandomLayerArgs {
   factor: number | [number, number];
   interpolation?: InterpolationType; // default = 'bilinear';
   seed?: number; // default = null;
   autoVectorize?: boolean;
}

const INTERPOLATION_KEYS = ['bilinear', 'nearest'] as const;
export const INTERPOLATION_METHODS = new Set(INTERPOLATION_KEYS);
type InterpolationType = typeof INTERPOLATION_KEYS[number];

/**
 * Preprocessing Layer with randomly varies image during training
 *
 * This layer randomly adjusts the width of a batch of images of a
 * batch of images by a random factor.
 *
 * The input should be a 3D (unbatched) or
 * 4D (batched) tensor in the `"channels_last"` image data format. Input pixel
 * values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and of integer
 * or floating point dtype. By default, the layer will output floats.
 *
 * tf methods implemented in tfjs: 'bilinear', 'nearest',
 * tf methods unimplemented in tfjs: 'bicubic', 'area', 'lanczos3', 'lanczos5',
 *                                   'gaussian', 'mitchellcubic'
 *
 */

export class RandomWidth extends BaseRandomLayer {
  /** @nocollapse */
  static override className = 'RandomWidth';
  private readonly factor: number | [number, number];
  private readonly interpolation?: InterpolationType;  // default = 'bilinear
  private widthLower: number;
  private widthUpper: number;
  private imgHeight: number;
  private widthFactor: Tensor<Rank.R1>;

  constructor(args: RandomWidthArgs) {
    super(args);
    const {factor, interpolation = 'bilinear'} = args;

    this.factor = factor;

    if (Array.isArray(this.factor) && this.factor.length === 2) {
      this.widthLower = this.factor[0];
      this.widthUpper = this.factor[1];
    } else if (!Array.isArray(this.factor) && this.factor > 0){
      this.widthLower = -this.factor;
      this.widthUpper = this.factor;
    } else {
      throw new ValueError(
        `Invalid factor: ${this.factor}. Must be positive number or tuple of 2 numbers`
      );
    }
    if (this.widthLower < -1.0 || this.widthUpper < -1.0) {
      throw new ValueError(
        `factor must have values larger than -1. Got: ${this.factor}`
      );
    }

    if (this.widthUpper < this.widthLower) {
      throw new ValueError(
        `factor cannot have upper bound less than lower bound.
        Got upper bound: ${this.widthUpper}.
        Got lower bound: ${this.widthLower}
      `);
    }

    if (interpolation) {
      if (INTERPOLATION_METHODS.has(interpolation)) {
        this.interpolation = interpolation;
      } else {
        throw new ValueError(`Invalid interpolation parameter: ${
            interpolation} is not implemented`);
      }
    } 
  }

  override getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      'factor': this.factor,
      'interpolation': this.interpolation,
    };

    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  override computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    const numChannels = inputShape[2];
    return [this.imgHeight, -1, numChannels];
  }

  override call(inputs: Tensor<Rank.R3>|Tensor<Rank.R4>,
    kwargs: Kwargs): Tensor[]|Tensor {

    return tidy(() => {
      const input = getExactlyOneTensor(inputs);
      this.imgHeight = input.shape[input.shape.length - 3];
      const imgWidth = input.shape[input.shape.length - 2];

      this.widthFactor = randomUniform([1],
        (1.0 + this.widthLower), (1.0 + this.widthUpper),
        'float32', this.randomGenerator.next()
      );

      let adjustedWidth = this.widthFactor.dataSync()[0] * imgWidth;
      adjustedWidth = Math.round(adjustedWidth);

      const size:[number, number] = [this.imgHeight, adjustedWidth];

      switch (this.interpolation) {
        case 'bilinear':
          return image.resizeBilinear(inputs, size);
        case 'nearest':
          return image.resizeNearestNeighbor(inputs, size);
        default:
          throw new Error(`Interpolation is ${this.interpolation}
          but only ${[...INTERPOLATION_METHODS]} are supported`);
      }
    });
  }
}

serialization.registerClass(RandomWidth);
