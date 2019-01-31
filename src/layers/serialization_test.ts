/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {serialization} from '@tensorflow/tfjs-core';

import {Initializer, Ones, Zeros} from '../initializers';
import {deserialize} from './serialization';

describe('Deserialization', () => {
  it('Zeros Initialzer', () => {
    const config: serialization.ConfigDict = {};
    config.className = 'Zeros';
    config.config = {};
    const initializer: Zeros = deserialize(config) as Initializer;
    expect(initializer instanceof (Zeros)).toEqual(true);
  });
  it('Ones Initialzer', () => {
    const config: serialization.ConfigDict = {};
    config.className = 'Ones';
    config.config = {};
    const initializer: Ones = deserialize(config) as Initializer;
    expect(initializer instanceof (Ones)).toEqual(true);
  });
});
