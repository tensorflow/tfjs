/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

 import { describeMathCPUAndGPU } from '../utils/test_utils';
 import { RandomGenerator } from '../backend/random_generator';

const randomGenerator = new RandomGenerator(42);

describeMathCPUAndGPU('RandomGenerator', () => {
   it('with seed!=null works and returns different seeds.', () => {
     const firstSeed = randomGenerator.next();
     const secondSeed = randomGenerator.next();
     expect(firstSeed).not.toEqual(secondSeed);
   });
 });
