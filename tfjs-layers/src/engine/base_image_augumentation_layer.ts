/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { BaseRandomLayer, BaseRandomLayerArgs } from './base_random_layer';
import { ValueError } from '../errors';

export declare interface BaseImageAugmentationLayerArgs extends BaseRandomLayerArgs {
  rate: number;
};

export abstract class BaseImageAugumentationLayer extends BaseRandomLayer {
/**
 * Abstract base layer for image augmentation.
 * This layer contains base functionalities for preprocessing layers which
 * augment image related data, eg. image and in future, label and bounding
 * boxes. The subclasses could avoid making certain mistakes and reduce code
 * duplications.
 *
 */

  static override className = 'BaseImageAugmentationLayer';
  private readonly rate: number;
  private seed?: number;

  constructor(args: BaseImageAugmentationLayerArgs) {
    super(args);
    this.rate = args.rate;

    if(args.seed) {
      this.seed = args.seed;
    } else {
      this.seed = null;
    }

  }

  autoVectorize() {
   return this.hasOwnProperty("autoVectorize") ? this.autoVectorize : true
  }
}
