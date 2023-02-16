/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { LayerArgs, Layer } from './topology';
import { RandomSeed } from '../backend/random_seed';
import { serialization } from '@tensorflow/tfjs-core';

export declare interface BaseRandomLayerArgs extends LayerArgs {
  seed?: number;
}

export abstract class BaseRandomLayer extends Layer {
  // A layer handle the random number creation and savemodel behavior.
  /** @nocollapse */
  static className = 'BaseRandomLayer';
  protected randomGenerator: RandomSeed;
  private seed?: number;

  constructor(args: BaseRandomLayerArgs) {
    super(args);
    this.seed = args.seed;
    this.randomGenerator = new RandomSeed(this.seed);
  }

  override getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      'seed': this.seed
    };

    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
