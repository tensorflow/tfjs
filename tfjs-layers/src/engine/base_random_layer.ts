/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { LayerArgs, Layer } from './topology';
import { randomGamma, randomNormal} from '@tensorflow/tfjs-core';
import { randomStandardNormal, randomUniform } from '@tensorflow/tfjs-core'
import { ValueError } from '../errors';

export declare interface BaseRandomLayerArgs extends LayerArgs {
  seed?: number;
  forceGenerator: boolean;
  rngType: string
}

export type RNGTypes = {
  [key: string]: Function;
};

export abstract class BaseRandomLayer extends Layer {
  // A layer handle the random number creation and savemodel behavior.
  // private readonly seed?: number;
  static className = 'RandomWidth';
  private randomGenerator: Function;
  private readonly rngType: string;

  private readonly rng_types: RNGTypes = {
    gamma: randomGamma,
    normal: randomNormal,
    standardNormal: randomStandardNormal,
    uniform: randomUniform
  };

  constructor(args: BaseRandomLayerArgs) {
    super(args);
    this.rngType = args.rngType;


    if (this.rng_types.hasOwnProperty(this.rngType)) {
      this.randomGenerator = this.rng_types[this.rngType]
    } else {
      throw new ValueError (
        `Invalid rngType provided.
        Received rngType=${this.rngType}`);
    }
  }
}
