/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Keeps track of seed and handles pseudorandomness
 * Instance created in BaseRandomLayer class
 * Utilized for random preprocessing layers
 */

export class RandomSeed {
  static className = 'RandomSeed';
  seed: number | undefined;
  constructor(seed: number | undefined) { 
    this.seed = seed; 
  }
  next(): number | undefined { 
    if (this.seed === undefined) {
      return undefined;
    }
    return this.seed++; 
  }
}
