/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import * as tf from './index';
import {ALL_ENVS, describeWithFlags} from './jasmine_util';
import {complex, scalar, tensor2d} from './ops/ops';
import {inferShape} from './tensor_util_env';
import * as util from './util';

describe('Util', () => {
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

  it('Arrays shuffle randomly', () => {
    // Create 1000 numbers ordered
    const a = Array.apply(0, {length: 1000}).map(Number.call, Number).slice(1);
    const b = [].concat(a);  // copy ES5 style
    util.shuffle(a);
    expect(a).not.toEqual(b);
    expect(a.length).toEqual(b.length);
  });

  it('Multiple arrays shuffle together', () => {
    // Create 1000 numbers ordered
    const a = Array.apply(0, {length: 1000}).map(Number.call, Number).slice(1);
    const b = [].concat(a);  // copies
    const c = [].concat(a);
    util.shuffleCombo(a, b);
    expect(a).not.toEqual(c);
    expect(a).toEqual(b);
    expect(a.length).toEqual(c.length);
  });

  it('Is integer', () => {
    expect(util.isInt(0.5)).toBe(false);
    expect(util.isInt(1)).toBe(true);
  });

  it('Size to squarish shape (perfect square)', () => {
    expect(util.sizeToSquarishShape(9)).toEqual([3, 3]);
  });

  it('Size to squarish shape (prime number)', () => {
    expect(util.sizeToSquarishShape(11)).toEqual([4, 3]);
  });

  it('Size to squarish shape (almost square)', () => {
    expect(util.sizeToSquarishShape(35)).toEqual([6, 6]);
  });

  it('Size of 1 to squarish shape', () => {
    expect(util.sizeToSquarishShape(1)).toEqual([1, 1]);
  });

  it('infer shape single number', () => {
    expect(inferShape(4)).toEqual([]);
  });

  it('infer shape 1d array', () => {
    expect(inferShape([1, 2, 5])).toEqual([3]);
  });

  it('infer shape 2d array', () => {
    expect(inferShape([[1, 2, 5], [5, 4, 1]])).toEqual([2, 3]);
  });

  it('infer shape 3d array', () => {
    const a = [[[1, 2], [2, 3], [5, 6]], [[5, 6], [4, 5], [1, 2]]];
    expect(inferShape(a)).toEqual([2, 3, 2]);
  });

  it('infer shape 4d array', () => {
    const a = [
      [[[1], [2]], [[2], [3]], [[5], [6]]], [[[5], [6]], [[4], [5]], [[1], [2]]]
    ];
    expect(inferShape(a)).toEqual([2, 3, 2, 1]);
  });

  it('infer shape of typed array', () => {
    const a = new Float32Array([1, 2, 3, 4, 5]);
    expect(inferShape(a)).toEqual([5]);
  });

  it('infer shape of Uint8Array[], string tensor', () => {
    const a = [new Uint8Array([1, 2]), new Uint8Array([3, 4])];
    expect(inferShape(a, 'string')).toEqual([2]);
  });

  it('infer shape of Uint8Array[][], string tensor', () => {
    const a = [
      [new Uint8Array([1]), new Uint8Array([2])],
      [new Uint8Array([1]), new Uint8Array([2])]
    ];
    expect(inferShape(a, 'string')).toEqual([2, 2]);
  });

  it('infer shape of Uint8Array[][][], string tensor', () => {
    const a = [
      [[new Uint8Array([1, 2])], [new Uint8Array([2, 1])]],
      [[new Uint8Array([1, 2])], [new Uint8Array([2, 1])]]
    ];
    expect(inferShape(a, 'string')).toEqual([2, 2, 1]);
  });
});

describe('util.flatten', () => {
  it('nested number arrays', () => {
    expect(util.flatten([[1, 2, 3], [4, 5, 6]])).toEqual([1, 2, 3, 4, 5, 6]);
    expect(util.flatten([[[1, 2], [3, 4], [5, 6], [7, 8]]])).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8
    ]);
    expect(util.flatten([1, 2, 3, 4, 5, 6])).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('nested string arrays', () => {
    expect(util.flatten([['a', 'b'], ['c', [['d']]]])).toEqual([
      'a', 'b', 'c', 'd'
    ]);
    expect(util.flatten([['a', ['b']], ['c', [['d']], 'e']])).toEqual([
      'a', 'b', 'c', 'd', 'e'
    ]);
  });

  it('mixed TypedArray and number[]', () => {
    const data =
        [new Float32Array([1, 2]), 3, [4, 5, new Float32Array([6, 7])]];
    expect(util.flatten(data)).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it('nested Uint8Arrays, skipTypedArray=true', () => {
    const data = [
      [new Uint8Array([1, 2]), new Uint8Array([3, 4])],
      [new Uint8Array([5, 6]), new Uint8Array([7, 8])]
    ];
    expect(util.flatten(data, [], true)).toEqual([
      new Uint8Array([1, 2]), new Uint8Array([3, 4]), new Uint8Array([5, 6]),
      new Uint8Array([7, 8])
    ]);
  });
});

function encodeStrings(a: string[]): Uint8Array[] {
  return a.map(s => util.encodeString(s));
}

describe('util.bytesFromStringArray', () => {
  it('count bytes after utf8 encoding', () => {
    expect(util.bytesFromStringArray(encodeStrings(['a', 'bb', 'ccc'])))
        .toBe(6);
    expect(util.bytesFromStringArray(encodeStrings(['a', 'bb', 'cccddd'])))
        .toBe(9);
    expect(util.bytesFromStringArray(encodeStrings(['даниел']))).toBe(6 * 2);
  });
});

describe('util.inferDtype', () => {
  it('a single string => string', () => {
    expect(util.inferDtype('hello')).toBe('string');
  });

  it('a single boolean => bool', () => {
    expect(util.inferDtype(true)).toBe('bool');
    expect(util.inferDtype(false)).toBe('bool');
  });

  it('a single number => float32', () => {
    expect(util.inferDtype(0)).toBe('float32');
    expect(util.inferDtype(34)).toBe('float32');
  });

  it('a list of strings => string', () => {
    // Flat.
    expect(util.inferDtype(['a', 'b', 'c'])).toBe('string');
    // Nested.
    expect(util.inferDtype([
      [['a']], [['b']], [['c']], [['d']]
    ])).toBe('string');
  });

  it('a list of bools => float32', () => {
    // Flat.
    expect(util.inferDtype([false, true, false])).toBe('bool');
    // Nested.
    expect(util.inferDtype([
      [[true]], [[false]], [[true]], [[true]]
    ])).toBe('bool');
  });

  it('a list of numbers => float32', () => {
    // Flat.
    expect(util.inferDtype([0, 1, 2])).toBe('float32');
    // Nested.
    expect(util.inferDtype([[[0]], [[1]], [[2]], [[3]]])).toBe('float32');
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

describe('util parseAxisParam', () => {
  it('axis=null returns no axes for scalar', () => {
    const axis: number = null;
    const shape: number[] = [];
    expect(util.parseAxisParam(axis, shape)).toEqual([]);
  });

  it('axis=null returns 0 axis for Tensor1D', () => {
    const axis: number = null;
    const shape = [4];
    expect(util.parseAxisParam(axis, shape)).toEqual([0]);
  });

  it('axis=null returns all axes for Tensor3D', () => {
    const axis: number[] = null;
    const shape = [3, 1, 2];
    expect(util.parseAxisParam(axis, shape)).toEqual([0, 1, 2]);
  });

  it('axis as a single number', () => {
    const axis = 1;
    const shape = [3, 1, 2];
    expect(util.parseAxisParam(axis, shape)).toEqual([1]);
  });

  it('axis as single negative number', () => {
    const axis = -1;
    const shape = [3, 1, 2];
    expect(util.parseAxisParam(axis, shape)).toEqual([2]);

    const axis2 = -2;
    expect(util.parseAxisParam(axis2, shape)).toEqual([1]);

    const axis3 = -3;
    expect(util.parseAxisParam(axis3, shape)).toEqual([0]);
  });

  it('axis as list of negative numbers', () => {
    const axis = [-1, -3];
    const shape = [3, 1, 2];
    expect(util.parseAxisParam(axis, shape)).toEqual([2, 0]);
  });

  it('axis as list of positive numbers', () => {
    const axis = [0, 2];
    const shape = [3, 1, 2];
    expect(util.parseAxisParam(axis, shape)).toEqual([0, 2]);
  });

  it('axis as combo of positive and negative numbers', () => {
    const axis = [0, -1];
    const shape = [3, 1, 2];
    expect(util.parseAxisParam(axis, shape)).toEqual([0, 2]);
  });

  it('axis out of range throws error', () => {
    const axis = -4;
    const shape = [3, 1, 2];
    expect(() => util.parseAxisParam(axis, shape)).toThrowError();

    const axis2 = 4;
    expect(() => util.parseAxisParam(axis2, shape)).toThrowError();
  });

  it('axis a list with one number out of range throws error', () => {
    const axis = [0, 4];
    const shape = [3, 1, 2];
    expect(() => util.parseAxisParam(axis, shape)).toThrowError();
  });

  it('axis with decimal value throws error', () => {
    const axis = 0.5;
    const shape = [3, 1, 2];
    expect(() => util.parseAxisParam(axis, shape)).toThrowError();
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
    it('should only reduce dimensions specified by negative axis', () => {
      const {newShape, keptDims} = util.squeezeShape([1, 1, 1, 1, 4], [-2, -3]);
      expect(newShape).toEqual([1, 1, 4]);
      expect(keptDims).toEqual([0, 1, 4]);
    });
    it('should only reduce dimensions specified by negative axis', () => {
      const axis = [-2, -3];
      util.squeezeShape([1, 1, 1, 1, 4], axis);
      expect(axis).toEqual([-2, -3]);
    });
    it('throws error when specified axis is not squeezable', () => {
      expect(() => util.squeezeShape([1, 1, 2, 1, 4], [1, 2])).toThrowError();
    });
    it('throws error when specified negative axis is not squeezable', () => {
      expect(() => util.squeezeShape([1, 1, 2, 1, 4], [-1, -2])).toThrowError();
    });
    it('throws error when specified axis is out of range', () => {
      expect(() => util.squeezeShape([1, 1, 2, 1, 4], [11, 22])).toThrowError();
    });
    it('throws error when specified negative axis is out of range', () => {
      expect(() => util.squeezeShape([1, 1, 2, 1, 4], [
        -11, -22
      ])).toThrowError();
    });
  });
});

describe('util.checkConversionForErrors', () => {
  it('Float32Array has NaN', () => {
    expect(
        () => util.checkConversionForErrors(
            new Float32Array([1, 2, 3, NaN, 4, 255]), 'float32'))
        .toThrowError();
  });

  it('Float32Array has Infinity', () => {
    expect(
        () => util.checkConversionForErrors(
            new Float32Array([1, 2, 3, Infinity, 4, 255]), 'float32'))
        .toThrowError();
  });

  it('Int32Array has NaN', () => {
    expect(() => util.checkConversionForErrors([1, 2, 3, 4, NaN], 'int32'))
        .toThrowError();
  });
});

describe('util.hasEncodingLoss', () => {
  it('complex64 to any', () => {
    expect(util.hasEncodingLoss('complex64', 'complex64')).toBe(false);
    expect(util.hasEncodingLoss('complex64', 'float32')).toBe(true);
    expect(util.hasEncodingLoss('complex64', 'int32')).toBe(true);
    expect(util.hasEncodingLoss('complex64', 'bool')).toBe(true);
  });

  it('any to complex64', () => {
    expect(util.hasEncodingLoss('bool', 'complex64')).toBe(false);
    expect(util.hasEncodingLoss('int32', 'complex64')).toBe(false);
    expect(util.hasEncodingLoss('float32', 'complex64')).toBe(false);
    expect(util.hasEncodingLoss('complex64', 'complex64')).toBe(false);
  });

  it('any to float32', () => {
    expect(util.hasEncodingLoss('bool', 'float32')).toBe(false);
    expect(util.hasEncodingLoss('int32', 'float32')).toBe(false);
    expect(util.hasEncodingLoss('float32', 'float32')).toBe(false);
    expect(util.hasEncodingLoss('complex64', 'float32')).toBe(true);
  });

  it('float32 to any', () => {
    expect(util.hasEncodingLoss('float32', 'float32')).toBe(false);
    expect(util.hasEncodingLoss('float32', 'int32')).toBe(true);
    expect(util.hasEncodingLoss('float32', 'bool')).toBe(true);
    expect(util.hasEncodingLoss('float32', 'complex64')).toBe(false);
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

describeWithFlags('util.toNestedArray', ALL_ENVS, () => {
  it('2 dimensions', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    expect(util.toNestedArray([2, 3], a)).toEqual([[1, 2, 3], [4, 5, 6]]);
  });

  it('3 dimensions (2x2x3)', () => {
    const a = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    expect(util.toNestedArray([2, 2, 3], a)).toEqual([
      [[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]
    ]);
  });

  it('3 dimensions (3x2x2)', () => {
    const a = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    expect(util.toNestedArray([3, 2, 2], a)).toEqual([
      [[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]
    ]);
  });

  it('invalid dimension', () => {
    const a = new Float32Array([1, 2, 3]);
    expect(() => util.toNestedArray([2, 2], a)).toThrowError();
  });

  it('tensor to nested array', async () => {
    const x = tensor2d([1, 2, 3, 4], [2, 2]);
    expect(util.toNestedArray(x.shape, await x.data())).toEqual([
      [1, 2], [3, 4]
    ]);
  });

  it('scalar to nested array', async () => {
    const x = scalar(1);
    expect(util.toNestedArray(x.shape, await x.data())).toEqual(1);
  });

  it('tensor with zero shape', () => {
    const a = new Float32Array([0, 1]);
    expect(util.toNestedArray([1, 0, 2], a)).toEqual([]);
  });
});

describeWithFlags('util.toNestedArray for a complex tensor', ALL_ENVS, () => {
  it('2 dimensions', () => {
    const a = new Float32Array([1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16]);
    expect(util.toNestedArray([2, 3], a, true)).toEqual([
      [1, 11, 2, 12, 3, 13], [4, 14, 5, 15, 6, 16]
    ]);
  });

  it('3 dimensions (2x2x3)', () => {
    const a = new Float32Array([
      0, 50, 1, 51, 2, 52, 3, 53, 4,  54, 5,  55,
      6, 56, 7, 57, 8, 58, 9, 59, 10, 60, 11, 61
    ]);
    expect(util.toNestedArray([2, 2, 3], a, true)).toEqual([
      [[0, 50, 1, 51, 2, 52], [3, 53, 4, 54, 5, 55]],
      [[6, 56, 7, 57, 8, 58], [9, 59, 10, 60, 11, 61]]
    ]);
  });

  it('3 dimensions (3x2x2)', () => {
    const a = new Float32Array([
      0, 50, 1, 51, 2, 52, 3, 53, 4,  54, 5,  55,
      6, 56, 7, 57, 8, 58, 9, 59, 10, 60, 11, 61
    ]);
    expect(util.toNestedArray([3, 2, 2], a, true)).toEqual([
      [[0, 50, 1, 51], [2, 52, 3, 53]], [[4, 54, 5, 55], [6, 56, 7, 57]],
      [[8, 58, 9, 59], [10, 60, 11, 61]]
    ]);
  });

  it('invalid dimension', () => {
    const a = new Float32Array([1, 11, 2, 12, 3, 13]);
    expect(() => util.toNestedArray([2, 2], a, true)).toThrowError();
  });

  it('tensor to nested array', async () => {
    const x = complex([[1, 2], [3, 4]], [[11, 12], [13, 14]]);
    expect(util.toNestedArray(x.shape, await x.data(), true)).toEqual([
      [1, 11, 2, 12], [3, 13, 4, 14]
    ]);
  });
});

describe('util.fetch', () => {
  it('should call the platform fetch', () => {
    spyOn(tf.env().platform, 'fetch').and.callFake(() => {});

    util.fetch('test/path', {method: 'GET'});

    expect(tf.env().platform.fetch).toHaveBeenCalledWith('test/path', {
      method: 'GET'
    });
  });
});

describe('util.encodeString', () => {
  it('Encode an empty string, default encoding', () => {
    const res = util.encodeString('');
    expect(res).toEqual(new Uint8Array([]));
  });

  it('Encode an empty string, utf-8 encoding', () => {
    const res = util.encodeString('', 'utf-8');
    expect(res).toEqual(new Uint8Array([]));
  });

  it('Encode an empty string, invalid decoding', () => {
    expect(() => util.encodeString('', 'foobarbax')).toThrowError();
  });

  it('Encode cyrillic letters', () => {
    const res = util.encodeString('Kaкo стe');
    expect(res).toEqual(
        new Uint8Array([75, 97, 208, 186, 111, 32, 209, 129, 209, 130, 101]));
  });

  it('Encode ascii letters', () => {
    const res = util.encodeString('hello');
    expect(res).toEqual(new Uint8Array([104, 101, 108, 108, 111]));
  });
});

describe('util.decodeString', () => {
  it('decode an empty string', () => {
    const s = util.decodeString(new Uint8Array([]));
    expect(s).toEqual('');
  });

  it('decode ascii', () => {
    const s = util.decodeString(new Uint8Array([104, 101, 108, 108, 111]));
    expect(s).toEqual('hello');
  });

  it('decode cyrillic', () => {
    const s = util.decodeString(
        new Uint8Array([75, 97, 208, 186, 111, 32, 209, 129, 209, 130, 101]));
    expect(s).toEqual('Kaкo стe');
  });

  it('decode utf-16', () => {
    const s = util.decodeString(
        new Uint8Array([255, 254, 237, 139, 0, 138, 4, 89, 6, 116]), 'utf-16');

    // UTF-16 allows optional presence of byte-order-mark (BOM)
    // Construct string for '语言处理', with and without BOM
    const expected = String.fromCodePoint(0x8bed, 0x8a00, 0x5904, 0x7406);
    const expectedBOM =
        String.fromCodePoint(0xfeff, 0x8bed, 0x8a00, 0x5904, 0x7406);

    if (s.codePointAt(0) === 0xfeff) {
      expect(s).toEqual(expectedBOM);
    } else {
      expect(s).toEqual(expected);
    }
  });

  it('assert promise', () => {
    const promise = new Promise(() => {});
    expect(util.isPromise(promise)).toBeTruthy();
    const promise2 = {then: () => {}};
    expect(util.isPromise(promise2)).toBeTruthy();
    const promise3 = {};
    expect(util.isPromise(promise3)).toBeFalsy();
  });
});
