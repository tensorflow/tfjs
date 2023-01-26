/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { image, Rank, serialization, Tensor, cast, stack, tidy } from '@tensorflow/tfjs-core';
import { getExactlyOneTensor } from '../../utils/types_utils';
import * as K from '../../backend/tfjs_backend';
import { Kwargs } from 'tfjs-layers/src/types';
import { ValueError } from 'tfjs-layers/src/errors';
import { BaseRandomLayerArgs, BaseRandomLayer } from 'tfjs-layers/src/engine/base_random_layer';
import * as tf from "@tensorflow/tfjs"

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

const INTERPOLATION_KEYS = ['bilinear', 'nearest'] as const;
const INTERPOLATION_METHODS = new Set(INTERPOLATION_KEYS);
type InterpolationType = typeof INTERPOLATION_KEYS[number];

export class RandomWidth extends BaseRandomLayer {
  /** @nocollapse */
  static override className = 'RandomWidth';
  private readonly factor: number | [number, number];
  private readonly interpolation?: InterpolationType;  // defualt = 'bilinear
  private readonly seed?: number; // default null
  private readonly autoVectorize?: boolean; // default false
  private widthLower: number;
  private widthUpper: number;
  private interpolationMethod: Function;

  constructor(args: RandomWidthArgs) {
    super(args);

    this.factor = args.factor;

    if (Array.isArray(this.factor) && this.factor.length === 2) { // should these values also be >= 0, only seeing bounds for single value vs array?
      this.widthLower = this.factor[0];
      this.widthUpper = this.factor[1];
    } else if (!Array.isArray(this.factor) && this.factor > 0){ //do these values need to be positive or can they equal 0?
      this.widthLower = -this.factor;
      this.widthUpper = this.factor;
    } else {
      throw new ValueError(`
      Invalid factor parameter: ${this.factor}.
      Must be positive number or tuple of 2 numbers`);
    }

    if (this.widthUpper < this.widthLower) {
      throw new ValueError(
        `factor cannot have upper bound less than lower bound.
        Got upper bound: ${this.widthUpper}.
        Got lower bound: ${this.widthLower}
      `)
    }

    if (this.widthLower < -1.0 || this.widthUpper < -1.0) { // does this logic conflict with line 68 & 65?
      throw new ValueError(
        `factor must have values larger than -1.
        Got: ${this.factor}`
      )
    }

    this.interpolation = args.interpolation;

    if (args.interpolation) {
      if (INTERPOLATION_METHODS.has(args.interpolation)) {
        this.interpolation = args.interpolation;
      } else {
        throw new ValueError(`Invalid interpolation parameter: ${
            args.interpolation} is not implemented`);
      }
    } else {
      this.interpolation = 'bilinear';
    }

    if (this.interpolation === 'bilinear') {
      this.interpolationMethod = image.resizeBilinear;
    } else if (this.interpolation === 'nearest') {
      this.interpolationMethod = image.resizeNearestNeighbor;
    } else {
      throw new Error(`Interpolation is ${this.interpolation} but only
      ${[...INTERPOLATION_METHODS]} are supported`);
    }
    if(args.seed) {
      this.seed = args.seed;
    } else {
      this.seed = null;
    }

    if(args.autoVectorize) {
      this.autoVectorize = args.autoVectorize;
    } else {
      this.autoVectorize = false;
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

//// New Code: vvvv

  override call(inputs: Tensor<Rank.R3>|Tensor<Rank.R4>,
    kwargs: Kwargs): Tensor[]|Tensor {


    return tidy(() => {
      if (kwargs.training) {
        // Inputs width-adjusted with random ops.
        const input = getExactlyOneTensor(inputs);
        const inputShape = input.shape;
        const imgHeight = inputShape.length - 3;
        const imgWidth = inputShape.length - 2;


        const widthFactor = super.randomGenerator(inputShape,
          (1.0 + this.widthLower), (1.0 + this.widthUpper)
        );
        const adjustedWidth = cast(widthFactor * imgWidth, 'int32');
        // const adjustedSize [imgHeight, adjustedWidth]; //This must be Tensor
        const adjustedSize = tf.tensor([imgHeight, adjustedWidth],[2],'int32');
        // const adjustedSize = tf.tensor([4, 7]); // this works,
        // but accepts only an array of 8-bit unsigned integers

        if (this.interpolation === 'bilinear') {
          return image.resizeBilinear(inputs, adjustedSize)

        } else if (this.interpolation === 'nearest') {
          return image.resizeNearestNeighbor(
              inputs, adjustedSize, !this.cropToAspectRatio);
        }

      } else {
        return inputs;
      }

    });
  }



}



  // override computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
  //   inputShape = getExactlyOneShape(inputShape);
  //   const numChannels = inputShape[2];
  //   return [this.height, this.width, numChannels];
  // }

serialization.registerClass(RandomWidth);

