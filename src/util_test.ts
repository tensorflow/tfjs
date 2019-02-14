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

import {inferShape} from './tensor_util_env';
import * as util from './util';
import {scalar, tensor2d} from './ops/ops';

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
      [[[1], [2]], [[2], [3]], [[5], [6]]],
      [[[5], [6]], [[4], [5]], [[1], [2]]]
    ];
    expect(inferShape(a)).toEqual([2, 3, 2, 1]);
  });

  it('infer shape of typed array', () => {
    const a = new Float32Array([1, 2, 3, 4, 5]);
    expect(inferShape(a)).toEqual([5]);
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
});

describe('util.bytesFromStringArray', () => {
  it('count each character as 2 bytes', () => {
    expect(util.bytesFromStringArray(['a', 'bb', 'ccc'])).toBe(6 * 2);
    expect(util.bytesFromStringArray(['a', 'bb', 'cccddd'])).toBe(9 * 2);
    expect(util.bytesFromStringArray(['даниел'])).toBe(6 * 2);
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

describe('util.checkComputationForErrors', () => {
  it('Float32Array has NaN', () => {
    expect(
        () => util.checkComputationForErrors(
            new Float32Array([1, 2, 3, NaN, 4, 255]), 'float32', ''))
        .toThrowError();
  });

  it('Float32Array has Infinity', () => {
    expect(
        () => util.checkComputationForErrors(
            new Float32Array([1, 2, 3, Infinity, 4, 255]), 'float32', ''))
        .toThrowError();
  });

  it('Float32Array no NaN', () => {
    // Int32 and Bool NaNs should not trigger an error.
    expect(
        () => util.checkComputationForErrors(
            new Float32Array([1, 2, 3, 4, -1, 255]), 'float32', ''))
        .not.toThrowError();
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

describe('util.toNestedArray', () => {
  it('2 dimensions', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    expect(util.toNestedArray([2, 3], a))
      .toEqual([[1,2,3], [4,5,6]]);
  });

  it('3 dimensions (2x2x3)', () => {
    const a = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    expect(util.toNestedArray([2, 2, 3], a))
      .toEqual([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]);
  });

  it('3 dimensions (3x2x2)', () => {
    const a = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    expect(util.toNestedArray([3, 2, 2], a))
      .toEqual([[[0, 1],[2, 3]],[[4, 5],[6, 7]],[[8, 9],[10, 11]]]);
  });

  it('invalid dimension', () => {
    const a = new Float32Array([1, 2, 3]);
    expect(() => util.toNestedArray([2, 2], a)).toThrowError();
  });

  it('tensor to nested array', () => {
    const x = tensor2d([1, 2, 3, 4], [2, 2]);
    expect(util.toNestedArray(x.shape, x.dataSync()))
      .toEqual([[1, 2], [3, 4]]);
  });

  it('scalar to nested array', () => {
    const x = scalar(1);
    expect(util.toNestedArray(x.shape, x.dataSync())).toEqual(1);
  });

  it('tensor with zero shape', () => {
    const a = new Float32Array([0, 1]);
    expect(util.toNestedArray([1, 0, 2], a)).toEqual([]);
  });
});

describe('util.monitorPromisesProgress', () => {
  it('Default progress from 0 to 1', (done) => {
    const expectFractions: number[] = [0.25, 0.50, 0.75, 1.00];
    const fractionList: number[] = [];
    const tasks = Array(4).fill(0).map(()=>{
      return Promise.resolve();
    });
    util.monitorPromisesProgress(tasks, (progress: number)=>{
      fractionList.push(parseFloat(progress.toFixed(2)));
    }).then(()=>{
      expect(fractionList).toEqual(expectFractions);
      done();
    });
  });

  it('Progress with pre-defined range', (done) => {
    const startFraction = 0.2;
    const endFraction = 0.8;
    const expectFractions: number[] = [0.35, 0.50, 0.65, 0.80];
    const fractionList: number[] = [];
    const tasks = Array(4).fill(0).map(()=>{
      return Promise.resolve();
    });
    util.monitorPromisesProgress(tasks, (progress: number)=>{
      fractionList.push(parseFloat(progress.toFixed(2)));
      }, startFraction, endFraction).then(()=>{
      expect(fractionList).toEqual(expectFractions);
      done();
    });
  });

  it('throws error when progress fraction is out of range', () => {
    expect(() => {
      const startFraction = -1;
      const endFraction = 1;
      const tasks = Array(4).fill(0).map(()=>{
        return Promise.resolve();
      });
      util.monitorPromisesProgress(tasks, (progress: number)=>{},
          startFraction, endFraction);
    }).toThrowError();
  });

  it('throws error when startFraction more than endFraction', () => {
    expect(() => {
      const startFraction = 0.8;
      const endFraction = 0.2;
      const tasks = Array(4).fill(0).map(()=>{
        return Promise.resolve();
      });
      util.monitorPromisesProgress(tasks, (progress: number)=>{},
          startFraction, endFraction);
    }).toThrowError();
  });

  it('throws error when promises is null', () => {
    expect(() => {
      util.monitorPromisesProgress(null, (progress: number)=>{});
    }).toThrowError();
  });

  it('throws error when promises is empty array', () => {
    expect(() => {
      util.monitorPromisesProgress([], (progress: number)=>{});
    }).toThrowError();
  });
});
