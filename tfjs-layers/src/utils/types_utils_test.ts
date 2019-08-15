/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as types_utils from './types_utils';

describe('isArrayOfShapes', () => {
  it('returns false for a single non-empty shape', () => {
    expect(types_utils.isArrayOfShapes([1, 2, 3])).toEqual(false);
  });
  it('returns false for a single empty shape', () => {
    expect(types_utils.isArrayOfShapes([])).toEqual(false);
  });
  it('returns true for an array of shapes', () => {
    expect(types_utils.isArrayOfShapes([[1], [2, 3]])).toEqual(true);
  });
  it('returns true for an array of shapes that includes empty shapes', () => {
    expect(types_utils.isArrayOfShapes([[], [2, 3]])).toEqual(true);
    expect(types_utils.isArrayOfShapes([[]])).toEqual(true);
    expect(types_utils.isArrayOfShapes([[], []])).toEqual(true);
  });
});

describe('normalizeShapeList', () => {
  it('returns an empty list if an empty list is passed in.', () => {
    expect(types_utils.normalizeShapeList([])).toEqual([]);
  });

  it('returns a list of shapes if a single shape is passed in.', () => {
    expect(types_utils.normalizeShapeList([1])).toEqual([[1]]);
  });

  it('returns a list of shapes if an empty shape is passed in.', () => {
    expect(types_utils.normalizeShapeList([[]])).toEqual([[]]);
  });

  it('returns a list of shapes if a list of shapes is passed in.', () => {
    expect(types_utils.normalizeShapeList([[1]])).toEqual([[1]]);
  });
});

describe('getExactlyOneShape', () => {
  it('single instance', () => {
    expect(types_utils.getExactlyOneShape([1, 2, 3])).toEqual([1, 2, 3]);
    expect(types_utils.getExactlyOneShape([null, 8])).toEqual([null, 8]);
    expect(types_utils.getExactlyOneShape([])).toEqual([]);
  });
  it('Array of length 1', () => {
    expect(types_utils.getExactlyOneShape([[1, 2]])).toEqual([1, 2]);
    expect(types_utils.getExactlyOneShape([[]])).toEqual([]);
  });
  it('Array of length 2: ValueError', () => {
    expect(() => types_utils.getExactlyOneShape([
      [1], [2]
    ])).toThrowError(/Expected exactly 1 Shape; got 2/);
  });
});
