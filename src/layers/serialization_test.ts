/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {Ones, Zeros} from '../initializers';
import {ConfigDict} from '../types';
import {deserialize} from './serialization';

describe('Deserialization', () => {
  it('Zeros Initialzer', () => {
    const config: ConfigDict = {className: 'Zeros', config: {}};
    const initializer: Zeros = deserialize(config);
    expect(initializer instanceof (Zeros)).toEqual(true);
  });
  it('Ones Initialzer', () => {
    const config: ConfigDict = {className: 'Ones', config: {}};
    const initializer: Ones = deserialize(config);
    expect(initializer instanceof (Ones)).toEqual(true);
  });
});
