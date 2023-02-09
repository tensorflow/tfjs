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
import { RandomGenerator } from '../backend/random_generator';

export declare interface BaseRandomLayerArgs extends LayerArgs {
  seed?: number;
}

export abstract class BaseRandomLayer extends Layer {
  // A layer handle the random number creation and savemodel behavior.
  /** @nocollapse */
  static className = 'BaseRandomLayer';
  protected randomGenerator: RandomGenerator;

  constructor(args: BaseRandomLayerArgs) {
    super(args);
    this.randomGenerator = new RandomGenerator(args.seed);
  }
}
