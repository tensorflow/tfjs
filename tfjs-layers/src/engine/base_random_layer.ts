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
  force_generator: boolean;
  rng_type: string
}

export type RNGTypes = {
  [key: string]: Function;
};

export abstract class BaseRandomLayer extends Layer {
  // A layer handle the random number creation and savemodel behavior.
  // private readonly seed?: number;
  static className = 'RandomWidth';
  private random_generator: Function;
  private readonly rng_type: string;

  private readonly rng_types: RNGTypes = {
    gamma: randomGamma,
    normal: randomNormal,
    standardNormal: randomStandardNormal,
    uniform: randomUniform
  };

  constructor(args: BaseRandomLayerArgs) {
    super(args);
    this.rng_type = args.rng_type;


    if (this.rng_types.hasOwnProperty(this.rng_type)) {
      this.random_generator = this.rng_types[this.rng_type]
    } else {
      throw new ValueError (
        `Invalid rng_type provided.
        Received rng_type=${this.rng_type}`);
    }
  }

}
