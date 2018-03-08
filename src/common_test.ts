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
 * Unit tests for common.ts.
 */

import {isValidTensorName} from './common';

describe('isValidTensorName', () => {
  it('Valid tensor names', () => {
    expect(isValidTensorName('a')).toEqual(true);
    expect(isValidTensorName('A')).toEqual(true);
    expect(isValidTensorName('foo1')).toEqual(true);
    expect(isValidTensorName('Foo2')).toEqual(true);
    expect(isValidTensorName('n_1')).toEqual(true);
    expect(isValidTensorName('n.1')).toEqual(true);
    expect(isValidTensorName('n_1_2')).toEqual(true);
    expect(isValidTensorName('n.1.2')).toEqual(true);
    expect(isValidTensorName('a/B/c')).toEqual(true);
    expect(isValidTensorName('z_1/z_2/z.3')).toEqual(true);
  });

  it('Invalid tensor names: empty', () => {
    expect(isValidTensorName('')).toEqual(false);
  });

  it('Invalid tensor names: whitespaces', () => {
    expect(isValidTensorName('a b')).toEqual(false);
    expect(isValidTensorName('ab ')).toEqual(false);
  });

  it('Invalid tensor names: forbidden characters', () => {
    expect(isValidTensorName('foo1-2')).toEqual(false);
    expect(isValidTensorName('bar3!4')).toEqual(false);
  });

  it('Invalid tensor names: invalid first characters', () => {
    expect(isValidTensorName('/foo/bar')).toEqual(false);
    expect(isValidTensorName('.baz')).toEqual(false);
    expect(isValidTensorName('_baz')).toEqual(false);
    expect(isValidTensorName('1Qux')).toEqual(false);
  });

  it('Invalid tensor names: non-ASCII', () => {
    expect(isValidTensorName('フ')).toEqual(false);
    expect(isValidTensorName('ξ')).toEqual(false);
  });
});
