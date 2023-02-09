/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { randomGamma, randomNormal} from '@tensorflow/tfjs-core';
import { randomStandardNormal, randomUniform } from '@tensorflow/tfjs-core';

type randomGammaType = typeof randomGamma;
type randomNormalType = typeof randomNormal;
type randomStandardNormalType = typeof randomStandardNormal;
type randomUniformType = typeof randomUniform;

export class RandomGenerator {
  /** @nocollapse */
  static className = 'RandomGenerator';
  protected currentSeed: number;
  private readonly seed: number;
  randomGamma: randomGammaType; 
  randomNormal: randomNormalType;
  randomStandardNormal: randomStandardNormalType;
  randomUniform: randomUniformType;
  
  constructor(seed: number) {
    this.seed = seed;
    this.currentSeed = seed;
    this.randomGamma = randomGamma;
    this.randomNormal = randomNormal;
    this.randomStandardNormal = randomStandardNormal;
    this.randomUniform = randomUniform;
  }

  next(): number | null {
    if (typeof this.seed === 'number'){
      return this.currentSeed++;
    }
    return null;
  }

  reset() {
    // Not sure if reset is even necessary
    this.currentSeed = this.seed;
  }
}
