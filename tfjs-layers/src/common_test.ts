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

import {checkDataFormat, checkInterpolationFormat, checkPaddingMode, checkPoolMode, getUniqueTensorName, isValidTensorName} from './common';
import {VALID_DATA_FORMAT_VALUES, VALID_INTERPOLATION_FORMAT_VALUES, VALID_PADDING_MODE_VALUES, VALID_POOL_MODE_VALUES} from './keras_format/common';

describe('checkDataFormat', () => {
  it('Valid values', () => {
    const extendedValues = VALID_DATA_FORMAT_VALUES.concat([undefined, null]);
    for (const validValue of extendedValues) {
      // Using implicit "expect().toNotThrow()" for valid values
      checkDataFormat(validValue);
    }
  });
  it('Invalid values', () => {
    // Test invalid values are rejected, and reported in the error.
    expect(() => checkDataFormat('foo')).toThrowError(/foo/);
    try {
      checkDataFormat('bad');
    } catch (e) {
      expect(e).toMatch('DataFormat');
      // Test that the error message contains the list of valid values.
      for (const validValue of VALID_DATA_FORMAT_VALUES) {
        expect(e).toMatch(validValue);
      }
    }
  });
});

describe('checkIntorpolationFormat', () => {
  it('Valid values', () => {
    const extendedValues =
        VALID_INTERPOLATION_FORMAT_VALUES.concat([undefined, null]);
    for (const validValue of extendedValues) {
      // Using implicit "expect().toNotThrow()" for valid values
      checkInterpolationFormat(validValue);
    }
  });
  it('Invalid values', () => {
    // Test invalid values are rejected, and reported in the error.
    expect(() => checkInterpolationFormat('foo')).toThrowError(/foo/);
    try {
      checkInterpolationFormat('bad');
    } catch (e) {
      expect(e).toMatch('InterpolationFormat');
      // Test that the error message contains the list of valid values.
      for (const validValue of VALID_INTERPOLATION_FORMAT_VALUES) {
        expect(e).toMatch(validValue);
      }
    }
  });
});
describe('checkPaddingMode', () => {
  it('Valid values', () => {
    const extendedValues = VALID_PADDING_MODE_VALUES.concat([undefined, null]);
    for (const validValue of extendedValues) {
      // Using implicit "expect().toNotThrow()" for valid values
      checkPaddingMode(validValue);
    }
  });
  it('Invalid values', () => {
    // Test invalid values are rejected, and reported in the error.
    expect(() => checkPaddingMode('foo')).toThrowError(/foo/);
    try {
      checkPaddingMode('bad');
    } catch (e) {
      expect(e).toMatch('PaddingMode');
      // Test that the error message contains the list of valid values.
      for (const validValue of VALID_PADDING_MODE_VALUES) {
        expect(e).toMatch(validValue);
      }
    }
  });
});

describe('checkPoolMode', () => {
  it('Valid values', () => {
    const extendedValues = VALID_POOL_MODE_VALUES.concat([undefined, null]);
    for (const validValue of extendedValues) {
      // Using implicit "expect().toNotThrow()" for valid values
      checkPoolMode(validValue);
    }
  });
  it('Invalid values', () => {
    // Test invalid values are rejected, and reported in the error.
    expect(() => checkPoolMode('foo')).toThrowError(/foo/);
    try {
      checkPoolMode('bad');
    } catch (e) {
      expect(e).toMatch('PoolMode');
      // Test that the error message contains the list of valid values.
      for (const validValue of VALID_POOL_MODE_VALUES) {
        expect(e).toMatch(validValue);
      }
    }
  });
});

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
    expect(isValidTensorName('z-1/z-2/z.3')).toEqual(true);
    expect(isValidTensorName('1Qux')).toEqual(true);
    expect(isValidTensorName('5-conv/kernel')).toEqual(true);
  });

  it('Invalid tensor names: empty', () => {
    expect(isValidTensorName('')).toEqual(false);
  });

  it('Invalid tensor names: whitespaces', () => {
    expect(isValidTensorName('a b')).toEqual(false);
    expect(isValidTensorName('ab ')).toEqual(false);
  });

  it('Invalid tensor names: forbidden characters', () => {
    expect(isValidTensorName('-foo1')).toEqual(false);
    expect(isValidTensorName('-foo2-')).toEqual(false);
    expect(isValidTensorName('bar3!4')).toEqual(false);
  });

  it('Invalid tensor names: invalid first characters', () => {
    expect(isValidTensorName('/foo/bar')).toEqual(false);
    expect(isValidTensorName('.baz')).toEqual(false);
    expect(isValidTensorName('_baz')).toEqual(false);
  });

  it('Invalid tensor names: non-ASCII', () => {
    expect(isValidTensorName('フ')).toEqual(false);
    expect(isValidTensorName('ξ')).toEqual(false);
  });
});

describe('getUniqueTensorName', () => {
  it('Adds unique suffixes to tensor names', () => {
    expect(getUniqueTensorName('xx')).toEqual('xx');
    expect(getUniqueTensorName('xx')).toEqual('xx_1');
    expect(getUniqueTensorName('xx')).toEqual('xx_2');
    expect(getUniqueTensorName('xx')).toEqual('xx_3');
  });

  it('Correctly handles preexisting unique suffixes on tensor names', () => {
    expect(getUniqueTensorName('yy')).toEqual('yy');
    expect(getUniqueTensorName('yy')).toEqual('yy_1');
    expect(getUniqueTensorName('yy_1')).toEqual('yy_1_1');
    expect(getUniqueTensorName('yy')).toEqual('yy_2');
    expect(getUniqueTensorName('yy_1')).toEqual('yy_1_2');
    expect(getUniqueTensorName('yy_2')).toEqual('yy_2_1');
    expect(getUniqueTensorName('yy')).toEqual('yy_3');
    expect(getUniqueTensorName('yy_1_1')).toEqual('yy_1_1_1');
  });
});
