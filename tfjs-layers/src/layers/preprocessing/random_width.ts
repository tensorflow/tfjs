/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { serialization, Tensor, mul, add, tidy } from '@tensorflow/tfjs-core';
import { getExactlyOneTensor } from '../../utils/types_utils';
import * as K from '../../backend/tfjs_backend';
import { Kwargs } from 'tfjs-layers/src/types';
import { ValueError } from 'tfjs-layers/src/errors';
import { BaseRandomLayerArgs, BaseRandomLayer } from 'tfjs-layers/src/engine/base_random_layer';

// export declare interface RandomWidthArgs extends BaseImageAugmentationLayerArgs {
export declare interface RandomWidthArgs extends BaseRandomLayerArgs {
   factor: number | [number, number];
   interpolation?: InterpolationType; // default = 'bilinear';
   seed?: number;// default = false;
   auto_vectorize?:boolean;
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

// export class RandomWidth extends BaseImageAugmentationLayer {

const INTERPOLATION_KEYS = ['bilinear', 'nearest'] as const;
const INTERPOLATION_METHODS = new Set(INTERPOLATION_KEYS);
type InterpolationType = typeof INTERPOLATION_KEYS[number];

export class RandomWidth extends BaseRandomLayer {
  /** @nocollapse */
  static override className = 'RandomWidth';
  private readonly factor: number | [number, number];
  private readonly interpolation?: InterpolationType;  // defualt = 'bilinear
  private readonly seed?: number; // default null
  private readonly auto_vectorize?: boolean; // default false
  private widthLower: number;
  private widthUpper: number;

  constructor(args: RandomWidthArgs) {
    super(args);

    this.factor = args.factor;

    if (Array.isArray(this.factor) && this.factor.length == 2) {
      this.widthLower = this.factor[0];
      this.widthUpper = this.factor[1];
    } else if (!Array.isArray(this.factor)){
      this.widthLower = -this.factor;
      this.widthUpper = this.factor;
    } else {
      throw new ValueError(`
      Invalid factor parameter: ${this.factor}.
      Must be array of size 2 or number`);
    }

    if (this.widthUpper < this.widthLower) {
      throw new ValueError(
        `factor cannot have upper bound less than lower bound.
        Got upper bound: ${this.widthUpper}.
        Got lower bound: ${this.widthLower}
      `)
    }

    if (this.widthLower < -1.0 || this.widthUpper < -1.0) {
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

   if(args.seed) {
       this.seed = args.seed;
   } else {
       this.seed = null;
   }

   if(args.auto_vectorize) {
    this.auto_vectorize = args.auto_vectorize;
   } else {
    this.auto_vectorize = false;
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

}

serialization.registerClass(RandomWidth);

