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

describe('toSnakeCase', () => {
  for (const [inputString, expectedOutput] of [
           ['', ''], ['A', 'a'], ['AA', 'aa'], ['AAA', 'aaa'], ['AAa', 'a_aa'],
           ['AA0', 'a_a0'], ['aB', 'a_b'], ['aBC', 'a_bc'], ['aBc', 'a_bc'],
           ['_', 'private_'], ['a', 'a'], ['_a', 'private_a']]) {
    it('creates expected output', () => {
      expect(utils.toSnakeCase(inputString)).toEqual(expectedOutput);
    });
  }
});

describe('toCamelCase', () => {
  for (const [inputString, expectedOutput] of [
           ['', ''], ['A', 'A'], ['aa', 'aa'], ['a_a', 'aA'],
           ['a_aa', 'aAa']]) {
    it('creates expected output', () => {
      expect(utils.toCamelCase(inputString)).toEqual(expectedOutput);
    });
  }
});

describe('stringsEqual', () => {
  it('null and undefined', () => {
    expect(utils.stringsEqual(null, null)).toEqual(true);
    expect(utils.stringsEqual(undefined, undefined)).toEqual(true);
    expect(utils.stringsEqual(undefined, null)).toEqual(false);
    expect(utils.stringsEqual(undefined, [])).toEqual(false);
    expect(utils.stringsEqual(null, [])).toEqual(false);
    expect(utils.stringsEqual(null, ['a'])).toEqual(false);
  });
  it('Empty arrays', () => {
    expect(utils.stringsEqual([], [])).toEqual(true);
    expect(utils.stringsEqual([], ['a'])).toEqual(false);
  });
  it('Non-empty arrays', () => {
    expect(utils.stringsEqual(['a', 'b', 'c', null], [
      'a', 'b', 'c', null
    ])).toEqual(true);
    expect(utils.stringsEqual(['a', 'b', 'c', ''], [
      'a', 'b', 'c', ''
    ])).toEqual(true);
    expect(utils.stringsEqual(['a', 'b', 'c', null], [
      'a', 'b', 'c', undefined
    ])).toEqual(false);
    expect(utils.stringsEqual(['a', 'b', 'c', ''], [
      'a', 'c', 'b', ''
    ])).toEqual(false);
  });
});

describe('unique', () => {
  it('null or undefined', () => {
    expect(utils.unique(null)).toEqual(null);
    expect(utils.unique(undefined)).toEqual(undefined);
  });
  it('empty array', () => {
    expect(utils.unique([])).toEqual([]);
  });
  it('Non-empty array: string', () => {
    expect(utils.unique(['foo', 'bar', 'foo'])).toEqual(['foo', 'bar']);
    expect(utils.unique(['foo', 'bar', ''])).toEqual(['foo', 'bar', '']);
    expect(utils.unique(['foo', 'bar', null, ''])).toEqual([
      'foo', 'bar', null, ''
    ]);
  });
  it('Non-empty array: number', () => {
    expect(utils.unique([1, 2, -1, 2])).toEqual([1, 2, -1]);
    expect(utils.unique([2, 3, 2, null])).toEqual([2, 3, null]);
  });
});

describe('isObjectEmpty', () => {
  it('null or undefined', () => {
    expect(() => utils.isObjectEmpty(null)).toThrowError();
    expect(() => utils.isObjectEmpty(undefined)).toThrowError();
  });
  it('empty object', () => {
    expect(utils.isObjectEmpty({})).toEqual(true);
  });
  it('Non-empty object', () => {
    expect(utils.isObjectEmpty({'a': 12})).toEqual(false);
    expect(utils.isObjectEmpty({'a': 12, 'b': 34})).toEqual(false);
  });
});

describe('checkArrayTypeAndLength', () => {
  it('checks types', () => {
    // [1,2,3] is made of all 'number's.
    expect(utils.checkArrayTypeAndLength([1, 2, 3], 'number')).toEqual(true);
    // ['hello', 'world'] is made of all 'strings's.
    expect(utils.checkArrayTypeAndLength(['hello', 'world'], 'string'))
        .toEqual(true);
    // [1,2,[3]] is not made of all 'number's.
    expect(utils.checkArrayTypeAndLength([1, 2, [3]], 'number')).toEqual(false);
  });
  it('checks lengths', () => {
    // length of [1,2,3] is >= 1.
    expect(utils.checkArrayTypeAndLength([1, 2, 3], 'number', 1)).toEqual(true);
    // length of [1,2,3] is >= 1 and <= 3.
    expect(utils.checkArrayTypeAndLength([1, 2, 3], 'number', 1, 3))
        .toEqual(true);
    // length of [1,2,3,4,5] is not >= 1 and <= 3.
    expect(utils.checkArrayTypeAndLength([1, 2, 3, 4, 5], 'number', 1, 3))
        .toEqual(false);
    // length of [1,2,3,4,5] is not >= 7 and <= 10.
    expect(utils.checkArrayTypeAndLength([1, 2, 3, 4, 5], 'number', 7, 10))
        .toEqual(false);
    // Length of the empty array is >= 0 and <= 0.
    expect(utils.checkArrayTypeAndLength([], 'does_not_matter', 0, 0))
        .toEqual(true);
  });
  it('rejects negative length limits', () => {
    expect(() => utils.checkArrayTypeAndLength([1, 2, 3], 'number', -1))
        .toThrowError();
  });
  it('rejects maxLength < minLength', () => {
    expect(() => utils.checkArrayTypeAndLength([1, 2, 3], 'number', 100, 2))
        .toThrowError();
  });
});

describe('formatValueAsFriendlyString', () => {
  it('null input', () => {
    expect(utils.formatAsFriendlyString(null)).toEqual('null');
  });

  it('string input', () => {
    expect(utils.formatAsFriendlyString('')).toEqual('\"\"');
    expect(utils.formatAsFriendlyString('foo')).toEqual('\"foo\"');
  });

  it('array input', () => {
    expect(utils.formatAsFriendlyString([])).toEqual('[]');
    expect(utils.formatAsFriendlyString([null, 3])).toEqual('[null,3]');
    expect(utils.formatAsFriendlyString([1, 3, 3, 7])).toEqual('[1,3,3,7]');
    expect(utils.formatAsFriendlyString([1, 3, 3, 'a'])).toEqual('[1,3,3,"a"]');
    expect(utils.formatAsFriendlyString([
      [1], 3, [3], 7
    ])).toEqual('[[1],3,[3],7]');
  });
});
