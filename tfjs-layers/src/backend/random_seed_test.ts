/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {RandomSeed} from './random_seed';
import {describeMathCPUAndGPU} from '../utils/test_utils';

describeMathCPUAndGPU('RandomSeed', () => {
  it('Checking if RandomSeed class handles pseudo randomness.', () => {
    const randomSeed = new RandomSeed(42);
    const firstSeed = randomSeed.seed;
    randomSeed.next();
    const secondSeed = randomSeed.seed;
    expect(firstSeed).not.toEqual(secondSeed);
  });
  it('Checking if RandomSeed class handles undefined seed.', () => {
    const randomSeed = new RandomSeed(undefined);
    const firstSeed = randomSeed.seed;
    const secondSeed = randomSeed.next();
    expect(firstSeed).toEqual(undefined);
    expect(secondSeed).toEqual(undefined);
  });
});
