/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as utils from './generic_utils';
import {pyNormalizeArrayIndex} from './generic_utils';

describe('pyListRepeat() ', () => {
  it('creates an empty array for 0 numValues', () => {
    expect(utils.pyListRepeat(null, 0)).toEqual([]);
  });

  it('creates an array with 1 value for 1 numValues', () => {
    const value = 'a';
    expect(utils.pyListRepeat(value, 1)).toEqual([value]);
  });

  it('creates an array with 3 values for 3 numValues', () => {
    const value = 'a';
    const numValues = 3;
    const expectedValue = [value, value, value];
    expect(utils.pyListRepeat(value, numValues)).toEqual(expectedValue);
  });

  it('throws an exception when numValues <0', () => {
    const fillFn = () => utils.pyListRepeat(null, -1);
    expect(fillFn).toThrowError();
  });

  it('takes an existing array and replicates its contents.', () => {
    const value = [1, 2];
    const numValues = 2;
    const expectedValue = [1, 2, 1, 2];
    expect(utils.pyListRepeat(value, numValues)).toEqual(expectedValue);
  });

  describe('pyGetAttr', () => {
    it('returns default value', () => {
      const expectedValue = 'a';
      const obj = {};
      expect(utils.pyGetAttr(obj, 'key', expectedValue)).toEqual(expectedValue);
    });

    it('returns assigned value if it exists', () => {
      const expectedValue = 'a';
      const defaultValue = 'b';
      const key = 'key';
      const obj = {key: expectedValue};
      expect(utils.pyGetAttr(obj, key, defaultValue)).toEqual(expectedValue);
    });

    it('throws AttributeError if attribute does not exist and no default ' +
           'value is provided.',
       () => {
         const obj = {};
         const key = 'key';
         const accessFn = () => {
           utils.pyGetAttr(obj, key);
         };
         expect(accessFn).toThrowError(/^pyGetAttr:/);
       });

    it('does not throw AttributeError if attribute does not exist and null ' +
           'value is provided as the default.',
       () => {
         const obj = {};
         const key = 'key';
         const defaultValue: string = null;
         expect(utils.pyGetAttr(obj, key, defaultValue)).toBeNull();
       });
  });
});

describe('pyNormalizeArrayIndex', () => {
  const x = [2, 2, 2];

  for (const index of [0, 1, 2]) {
    it('returns index if index >= 0 and index < x.length', () => {
      expect(pyNormalizeArrayIndex(x, index)).toEqual(index);
    });
  }

  for (const [index, expected] of[[-1, 2], [-2, 1], [-3, 0]]) {
    it('returns index if index < 0 and abs(index) <= x.length', () => {
      expect(pyNormalizeArrayIndex(x, index)).toEqual(expected);
    });
  }

  it('throws an exception if the array is null', () => {
    expect(() => pyNormalizeArrayIndex(null, 0))
        .toThrowError(/Must provide a valid array/);
  });

  it('throws an exception if the index is null', () => {
    expect(() => pyNormalizeArrayIndex(x, null))
        .toThrowError(/Must provide a valid array/);
  });

  for (const index of [3, -4]) {
    it('throws an exception if the index is out of range', () => {
      expect(() => pyNormalizeArrayIndex(x, index))
          .toThrowError(/Index.*out of range/);
    });
  }
});

describe('assert', () => {
  for (const x of [false, null, undefined]) {
    it('throws error for false conditions', () => {
      expect(() => utils.assert(x)).toThrowError();
    });
  }

  it('doesn\'t throw error for true conditions', () => {
    expect(() => utils.assert(true)).not.toThrowError();
  });
});

describe('count', () => {
  it('string array, non-empty', () => {
    const array: string[] = ['foo', 'bar', 'foo'];
    expect(utils.count(array, 'foo')).toEqual(2);
    expect(utils.count(array, 'bar')).toEqual(1);
    expect(utils.count(array, 'baz')).toEqual(0);
    expect(utils.count(array, '')).toEqual(0);
  });
  it('number array, non-empty', () => {
    const array: number[] = [-1, 1, 3, 3, 7, -1, 1.337, -1];
    expect(utils.count(array, 1)).toEqual(1);
    expect(utils.count(array, 3)).toEqual(2);
    expect(utils.count(array, 1.337)).toEqual(1);
    expect(utils.count(array, -1)).toEqual(3);
    expect(utils.count(array, 0)).toEqual(0);
  });
  it('string array, empty', () => {
    const array: string[] = [];
    expect(utils.count(array, 'foo')).toEqual(0);
    expect(utils.count(array, 'bar')).toEqual(0);
    expect(utils.count(array, 'baz')).toEqual(0);
    expect(utils.count(array, '')).toEqual(0);
  });
});

describe('Compare functions', () => {
  const inputs =
      [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]];

  for (const input of inputs) {
    it('cmp sorts numbers in ascending order', () => {
      const expected = [1, 2, 3];
      expect(input.slice().sort(utils.numberCompare)).toEqual(expected);
    });
  }

  for (const input of inputs) {
    it('reverseCmp sorts numbers in ascending order', () => {
      const expected = [3, 2, 1];
      expect(input.slice().sort(utils.reverseNumberCompare)).toEqual(expected);
    });
  }
});

describe('toList', () => {
  it('creates array from non-array.', () => {
    const value = 1;
    expect(utils.toList(value)).toEqual([value]);
  });

  it('returns array if passed an array.', () => {
    const value = [1];
    expect(utils.toList(value)).toEqual(value);
  });
});

describe('isArrayOfShapes', () => {
  it('returns false for a single non-empty shape', () => {
    expect(utils.isArrayOfShapes([1, 2, 3])).toEqual(false);
  });
  it('returns false for a single empty shape', () => {
    expect(utils.isArrayOfShapes([])).toEqual(false);
  });
  it('returns true for an array of shapes', () => {
    expect(utils.isArrayOfShapes([[1], [2, 3]])).toEqual(true);
  });
  it('returns true for an array of shapes that includes empty shapes', () => {
    expect(utils.isArrayOfShapes([[], [2, 3]])).toEqual(true);
    expect(utils.isArrayOfShapes([[]])).toEqual(true);
    expect(utils.isArrayOfShapes([[], []])).toEqual(true);
  });
});

describe('normalizeShapeList', () => {
  it('returns an empty list if an empty list is passed in.', () => {
    expect(utils.normalizeShapeList([])).toEqual([]);
  });

  it('returns a list of shapes if a single shape is passed in.', () => {
    expect(utils.normalizeShapeList([1])).toEqual([[1]]);
  });

  it('returns a list of shapes if an empty shape is passed in.', () => {
    expect(utils.normalizeShapeList([[]])).toEqual([[]]);
  });

  it('returns a list of shapes if a list of shapes is passed in.', () => {
    expect(utils.normalizeShapeList([[1]])).toEqual([[1]]);
  });
});

describe('isAllNull', () => {
  it('is true for empty lists.', () => {
    expect(utils.isAllNullOrUndefined([])).toEqual(true);
  });

  it('is true for lists with null.', () => {
    expect(utils.isAllNullOrUndefined([null])).toEqual(true);
  });

  it('is true for lists with undefined.', () => {
    expect(utils.isAllNullOrUndefined([undefined])).toEqual(true);
  });

  it('is false for lists with non-null.', () => {
    expect(utils.isAllNullOrUndefined([1])).toEqual(false);
  });

  it('is false for lists with non-null and null.', () => {
    expect(utils.isAllNullOrUndefined([null, 1])).toEqual(false);
  });
});

describe('toSnakeCase', () => {
  for (const [inputString, expectedOutput] of[
           ['', ''], ['A', 'a'], ['AA', 'aa'], ['AAA', 'aaa'], ['AAa', 'a_aa'],
           ['AA0', 'a_a0'], ['aB', 'a_b'], ['aBC', 'a_bc'], ['aBc', 'a_bc'],
           ['_', 'private_'], ['a', 'a'], ['_a', 'private_a']]) {
    it('creates expected output', () => {
      expect(utils.toSnakeCase(inputString)).toEqual(expectedOutput);
    });
  }
});

describe('toCamelCase', () => {
  for (const [inputString, expectedOutput] of[
           ['', ''], ['A', 'A'], ['aa', 'aa'], ['a_a', 'aA'],
           ['a_aa', 'aAa']]) {
    it('creates expected output', () => {
      expect(utils.toCamelCase(inputString)).toEqual(expectedOutput);
    });
  }
});

describe('SerializableEnumRegistry', () => {
  it('contains false if no registered converter for field name?', () => {
    expect(utils.SerializableEnumRegistry.contains('foo')).toEqual(false);
  });
  // Note: Somewhat unexpectedily? all the converters are regsitered already
  it('contains true if no registered converter for field name?', () => {
    expect(utils.SerializableEnumRegistry.contains('data_format'))
        .toEqual(true);
  });
  it('throws if you try to register a duplicate key?', () => {
    expect(() => {
      utils.SerializableEnumRegistry.register('data_format', {});
    }).toThrowError(/Attempting to register/);
  });
  it('returns null if no such value', () => {
    expect(utils.SerializableEnumRegistry.lookup('data_format', 'xxxx'))
        .toEqual(undefined);
  });
  it('correctly maps', () => {
    expect(
        utils.SerializableEnumRegistry.lookup('data_format', 'channels_first'))
        .toEqual('channelFirst');
  });
});

describe('getExactlyOneShape', () => {
  it('single instance', () => {
    expect(utils.getExactlyOneShape([1, 2, 3])).toEqual([1, 2, 3]);
    expect(utils.getExactlyOneShape([null, 8])).toEqual([null, 8]);
    expect(utils.getExactlyOneShape([])).toEqual([]);
  });
  it('Array of length 1', () => {
    expect(utils.getExactlyOneShape([[1, 2]])).toEqual([1, 2]);
    expect(utils.getExactlyOneShape([[]])).toEqual([]);
  });
  it('Array of length 2: ValueError', () => {
    expect(() => utils.getExactlyOneShape([
      [1], [2]
    ])).toThrowError(/Expected exactly 1 Shape; got 2/);
  });
});
