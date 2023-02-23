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
  currentSeed: number;
  constructor(readonly seed: number | undefined) { 
    this.currentSeed = seed; 
  }
  next(): number | undefined { 
    if (this.currentSeed === undefined) {
      return undefined;
    }
    return this.currentSeed++; 
  }
}
