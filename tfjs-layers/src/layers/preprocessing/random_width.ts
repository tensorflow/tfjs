/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { image, Rank, serialization, Tensor, tidy } from '@tensorflow/tfjs-core';
import { getExactlyOneTensor, getExactlyOneShape } from '../../utils/types_utils';
import {Shape} from '../../keras_format/common';
import { Kwargs } from 'tfjs-layers/src/types';
import { ValueError } from 'tfjs-layers/src/errors';
import { BaseRandomLayerArgs, BaseRandomLayer } from 'tfjs-layers/src/engine/base_random_layer';

// export declare interface RandomWidthArgs extends BaseImageAugmentationLayerArgs {
export declare interface RandomWidthArgs extends BaseRandomLayerArgs {
   factor: number | [number, number];
   interpolation?: InterpolationType; // default = 'bilinear';
   seed?: number;// default = false;
   autoVectorize?:boolean;
   rngType: string;
}

/**
 * Preprocessing Layer with randomly varies image during training
 *
 * This layer randomly adjusts the width of a batch of images of a
 * batch of images by a random factor.

 * The input should be a 3D (unbatched) or
 * 4D (batched) tensor in the `"channels_last"` image data format. Input pixel
 * values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and of interger
 * or floating point dtype. By default, the layer will output floats.
 *
 *tf methods implemented in tfjs: 'bilinear', 'nearest',
 * tf methods unimplemented in tfjs: 'bicubic', 'area', 'lanczos3', 'lanczos5',
 *                                   'gaussian', 'mitchellcubic'
 *
 */

// const INTERPOLATION_TODO = ["bicubic", "area", "lanczos3", "lanczos5",
//                             "gaussian", "mitchellcubic"]  as const;
const INTERPOLATION_KEYS = ['bilinear', 'nearest'] as const;
export const INTERPOLATION_METHODS = new Set(INTERPOLATION_KEYS);
type InterpolationType = typeof INTERPOLATION_KEYS[number];

export class RandomWidth extends BaseRandomLayer {
  /** @nocollapse */
  static override className = 'RandomWidth';
  private readonly factor: number | [number, number];
  private readonly interpolation?: InterpolationType;  // defualt = 'bilinear
  private readonly seed?: number; // default null
  private widthLower: number;
  private widthUpper: number;
  private imgHeight: number;
  private adjustedWidth: number;

  constructor(args: RandomWidthArgs) {
    super(args);
    this.factor = args.factor;

    if (Array.isArray(this.factor) && this.factor.length === 2) {
      this.widthLower = this.factor[0];
      this.widthUpper = this.factor[1];
    } else if (!Array.isArray(this.factor) && this.factor > 0){
      this.widthLower = -this.factor;
      this.widthUpper = this.factor;
    } else {
      throw new ValueError(`
      Invalid factor parameter: ${this.factor}.
      Must be positive number or tuple of 2 numbers`);
    }

// if was no error, then widthLower and widthUpper have values
    if (this.widthLower < -1.0 || this.widthUpper < -1.0) {
      // does this logic conflict with line 68 & 65?
      // line numbers were changed
      throw new ValueError(
        `factor must have values larger than -1.
        Got: ${this.factor}`
      )
    }

    if (this.widthUpper < this.widthLower) {
      throw new ValueError(
        `factor cannot have upper bound less than lower bound.
        Got upper bound: ${this.widthUpper}.
        Got lower bound: ${this.widthLower}
      `)
    }

    if (args.interpolation) {
      if (INTERPOLATION_METHODS.has(args.interpolation)) {
        this.interpolation = args.interpolation;
      } else {
        this.interpolation = 'bilinear';
      }
    }

    if(args.seed) {
      this.seed = args.seed;
    } else {
      this.seed = null;
    }
  }

  override getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      'factor': this.factor,
      'interpolation': this.interpolation,
      'seed': this.seed,
    };

    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  override computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    const numChannels = inputShape[2];
    return [this.imgHeight, this.adjustedWidth, numChannels];
  }


  override call(inputs: Tensor<Rank.R3>|Tensor<Rank.R4>,
    kwargs: Kwargs): Tensor[]|Tensor {


    return tidy(() => {
      if (kwargs.training) {
        // Inputs width-adjusted with random ops.
        const input = getExactlyOneTensor(inputs);
        const inputShape = input.shape;
        this.imgHeight = inputShape.length - 3;
        const imgWidth = inputShape.length - 2;

        const randomUniform = super.setRNGType('uniform');

        const widthFactor = randomUniform(inputShape,
          (1.0 + this.widthLower), (1.0 + this.widthUpper)
        );
        this.adjustedWidth = widthFactor * imgWidth

        if (this.interpolation === 'bilinear') {
          return image.resizeBilinear(inputs, [this.imgHeight, this.adjustedWidth])

        } else if (this.interpolation === 'nearest') {
          return image.resizeNearestNeighbor(inputs, [this.imgHeight, this.adjustedWidth]);
        } else {
          throw new Error(`Interpolation is ${this.interpolation}
          but only ${[...INTERPOLATION_METHODS]} are supported`);
        }

      } else {
        return inputs;
      }
    });
  }
}

serialization.registerClass(RandomWidth);

