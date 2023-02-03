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
import { randomStandardNormal, randomUniform } from '@tensorflow/tfjs-core';
import { ValueError } from '../errors';

export declare interface BaseRandomLayerArgs extends LayerArgs {}

export type RNGTypes = {
  [key: string]: Function;
};

export abstract class BaseRandomLayer extends Layer {
  // A layer handle the random number creation and savemodel behavior.
  /** @nocollapse */
  static className = 'RandomWidth';
  randomGenerator: Function;
  private rngType: string;

  private readonly rngTypes: RNGTypes = {
    gamma: randomGamma,
    normal: randomNormal,
    standardNormal: randomStandardNormal,
    uniform: randomUniform
  };

  constructor(args: BaseRandomLayerArgs) {
    super(args);
  }

  public setRNGType = (rngType: string) => {
    this.rngType = rngType;

    if (this.rngTypes.hasOwnProperty(this.rngType)) {
      this.randomGenerator = this.rngTypes[this.rngType];
    } else {
      throw new ValueError (
        `Invalid rngType provided.
        Received rngType=${this.rngType}`);
    }
    return this.randomGenerator;
  }
}
