/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as util from './util';

describe('Util', () => {
  it('Flatten arrays', () => {
    expect(util.flatten([[1, 2, 3], [4, 5, 6]])).toEqual([1, 2, 3, 4, 5, 6]);
    expect(util.flatten([[[1, 2], [3, 4], [5, 6], [7, 8]]])).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8
    ]);
    expect(util.flatten([1, 2, 3, 4, 5, 6])).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('Correctly gets size from shape', () => {
    expect(util.sizeFromShape([1, 2, 3, 4])).toEqual(24);
  });

  it('Correctly identifies scalars', () => {
    expect(util.isScalarShape([])).toBe(true);
    expect(util.isScalarShape([1, 2])).toBe(false);
    expect(util.isScalarShape([1])).toBe(false);
  });

  it('Number arrays equal', () => {
    expect(util.arraysEqual([1, 2, 3, 6], [1, 2, 3, 6])).toBe(true);
    expect(util.arraysEqual([1, 2], [1, 2, 3])).toBe(false);
    expect(util.arraysEqual([1, 2, 5], [1, 2])).toBe(false);
  });

  it('Is integer', () => {
    expect(util.isInt(0.5)).toBe(false);
    expect(util.isInt(1)).toBe(true);
  });

  it('Size to squarish shape (perfect square)', () => {
    expect(util.sizeToSquarishShape(9)).toEqual([3, 3]);
  });

  it('Size to squarish shape (prime number)', () => {
    expect(util.sizeToSquarishShape(11)).toEqual([1, 11]);
  });

  it('Size to squarish shape (almost square)', () => {
    expect(util.sizeToSquarishShape(35)).toEqual([5, 7]);
  });

  it('Size of 1 to squarish shape', () => {
    expect(util.sizeToSquarishShape(1)).toEqual([1, 1]);
  });

  it('infer shape single number', () => {
    expect(util.inferShape(4)).toEqual([]);
  });

  it('infer shape 1d array', () => {
    expect(util.inferShape([1, 2, 5])).toEqual([3]);
  });

  it('infer shape 2d array', () => {
    expect(util.inferShape([[1, 2, 5], [5, 4, 1]])).toEqual([2, 3]);
  });

  it('infer shape 3d array', () => {
    const a = [[[1, 2], [2, 3], [5, 6]], [[5, 6], [4, 5], [1, 2]]];
    expect(util.inferShape(a)).toEqual([2, 3, 2]);
  });

  it('infer shape 4d array', () => {
    const a = [
      [[[1], [2]], [[2], [3]], [[5], [6]]],
      [[[5], [6]], [[4], [5]], [[1], [2]]]
    ];
    expect(util.inferShape(a)).toEqual([2, 3, 2, 1]);
  });

  it('infer shape of typed array', () => {
    const a = new Float32Array([1, 2, 3, 4, 5]);
    expect(util.inferShape(a)).toEqual([5]);
  });
});

describe('util.repeatedTry', () => {
  it('resolves', (doneFn) => {
    let counter = 0;
    const checkFn = () => {
      counter++;
      if (counter === 2) {
        return true;
      }
      return false;
    };

    util.repeatedTry(checkFn).then(doneFn).catch(() => {
      throw new Error('Rejected backoff.');
    });
  });
  it('rejects', (doneFn) => {
    const checkFn = () => false;

    util.repeatedTry(checkFn, () => 0, 5)
        .then(() => {
          throw new Error('Backoff resolved');
        })
        .catch(doneFn);
  });
});

describe('util.inferFromImplicitShape', () => {
  it('empty shape', () => {
    const result = util.inferFromImplicitShape([], 0);
    expect(result).toEqual([]);
  });

  it('[2, 3, 4] -> [2, 3, 4]', () => {
    const result = util.inferFromImplicitShape([2, 3, 4], 24);
    expect(result).toEqual([2, 3, 4]);
  });

  it('[2, -1, 4] -> [2, 3, 4], size=24', () => {
    const result = util.inferFromImplicitShape([2, -1, 4], 24);
    expect(result).toEqual([2, 3, 4]);
  });

  it('[-1, 3, 4] -> [2, 3, 4], size=24', () => {
    const result = util.inferFromImplicitShape([-1, 3, 4], 24);
    expect(result).toEqual([2, 3, 4]);
  });

  it('[2, 3, -1] -> [2, 3, 4], size=24', () => {
    const result = util.inferFromImplicitShape([2, 3, -1], 24);
    expect(result).toEqual([2, 3, 4]);
  });

  it('[2, -1, -1] throws error', () => {
    expect(() => util.inferFromImplicitShape([2, -1, -1], 24)).toThrowError();
  });

  it('[2, 3, -1] size=13 throws error', () => {
    expect(() => util.inferFromImplicitShape([2, 3, -1], 13)).toThrowError();
  });

  it('[2, 3, 4] size=25 (should be 24) throws error', () => {
    expect(() => util.inferFromImplicitShape([2, 3, 4], 25)).toThrowError();
  });
});

describe('util.squeezeShape', () => {
  it('scalar', () => {
    const {newShape, keptDims} = util.squeezeShape([]);
    expect(newShape).toEqual([]);
    expect(keptDims).toEqual([]);
  });

  it('1x1 reduced to scalar', () => {
    const {newShape, keptDims} = util.squeezeShape([1, 1]);
    expect(newShape).toEqual([]);
    expect(keptDims).toEqual([]);
  });

  it('1x3x1 reduced to [3]', () => {
    const {newShape, keptDims} = util.squeezeShape([1, 3, 1]);
    expect(newShape).toEqual([3]);
    expect(keptDims).toEqual([1]);
  });

  it('1x1x4 reduced to [4]', () => {
    const {newShape, keptDims} = util.squeezeShape([1, 1, 4]);
    expect(newShape).toEqual([4]);
    expect(keptDims).toEqual([2]);
  });

  it('2x3x4 not reduction', () => {
    const {newShape, keptDims} = util.squeezeShape([2, 3, 4]);
    expect(newShape).toEqual([2, 3, 4]);
    expect(keptDims).toEqual([0, 1, 2]);
  });

  describe('with axis', () => {
    it('should only reduce dimensions specified by axis', () => {
      const {newShape, keptDims} = util.squeezeShape([1, 1, 1, 1, 4], [1, 2]);
      expect(newShape).toEqual([1, 1, 4]);
      expect(keptDims).toEqual([0, 3, 4]);
    });
    it('throws error when specified axis is not squeezable', () => {
      expect(() => util.squeezeShape([1, 1, 2, 1, 4], [1, 2])).toThrowError();
    });
  });
});

describe('util.checkComputationForNaN', () => {
  it('Float32Array has NaN', () => {
    expect(
        () => util.checkComputationForNaN(
            new Float32Array([1, 2, 3, NaN, 4, 255]), 'float32', ''))
        .toThrowError();
  });

  it('Float32Array no NaN', () => {
    // Int32 and Bool NaNs should not trigger an error.
    expect(
        () => util.checkComputationForNaN(
            new Float32Array([1, 2, 3, 4, -1, 255]), 'float32', ''))
        .not.toThrowError();
  });
});

describe('util.checkConversionForNaN', () => {
  // NaN is a valid value for type Float32
  it('Float32Array has NaN', () => {
    expect(
        () => util.checkConversionForNaN(
            new Float32Array([1, 2, 3, NaN, 4, 255]), 'float32'))
        .not.toThrowError();
  });
  // NaN should not be present in other types. Error should be thrown.
  it('Int32Array has NaN', () => {
    expect(() => util.checkConversionForNaN([1, 2, 3, 4, NaN], 'int32'))
        .toThrowError();
  });
});

describe('util.hasEncodingLoss', () => {
  it('any to float32', () => {
    expect(util.hasEncodingLoss('bool', 'float32')).toBe(false);
    expect(util.hasEncodingLoss('int32', 'float32')).toBe(false);
    expect(util.hasEncodingLoss('float32', 'float32')).toBe(false);
  });

  it('float32 to any', () => {
    expect(util.hasEncodingLoss('float32', 'float32')).toBe(false);
    expect(util.hasEncodingLoss('float32', 'int32')).toBe(true);
    expect(util.hasEncodingLoss('float32', 'bool')).toBe(true);
  });

  it('int32 to lower', () => {
    expect(util.hasEncodingLoss('int32', 'int32')).toBe(false);
    expect(util.hasEncodingLoss('int32', 'bool')).toBe(true);
  });

  it('lower to int32', () => {
    expect(util.hasEncodingLoss('bool', 'int32')).toBe(false);
  });

  it('bool to bool', () => {
    expect(util.hasEncodingLoss('bool', 'bool')).toBe(false);
  });
});
