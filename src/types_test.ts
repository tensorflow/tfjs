/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit tests for -specific types.
 */

// tslint:disable:max-line-length

import {SymbolicTensor} from './types';

// tslint:enable:max-line-length

/**
 * Unit tests for SymbolicTensor.
 */
describe('SymbolicTensor Test', () => {
  it('Correct dtype and shape properties', () => {
    const st1 = new SymbolicTensor('float32', [4, 6], null, [], {});
    expect(st1.dtype).toEqual('float32');
    expect(st1.shape).toEqual([4, 6]);
  });

  it('Correct names and ids', () => {
    const st1 = new SymbolicTensor(
        'float32', [2, 2], null, [], {}, 'TestSymbolicTensor');
    const st2 = new SymbolicTensor(
        'float32', [2, 2], null, [], {}, 'TestSymbolicTensor');
    expect(st1.name.indexOf('TestSymbolicTensor')).toEqual(0);
    expect(st2.name.indexOf('TestSymbolicTensor')).toEqual(0);
    // Explicit names of symbolic tensors should be unique.
    expect(st1 === st2).toBe(false);

    expect(st1.id).toBeGreaterThanOrEqual(0);
    expect(st2.id).toBeGreaterThanOrEqual(0);
    expect(st1.id === st2.id).toBe(false);
  });

  it('Invalid tensor name leads to error', () => {
    expect(() => new SymbolicTensor('float32', [2, 2], null, [], {}, '!'))
        .toThrowError();
  });
});
