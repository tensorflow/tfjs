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
});

describe('util.getBroadcastedShape', () => {
  it('two scalars', () => {
    const res = util.assertAndGetBroadcastedShape([], []);
    expect(res).toEqual([]);
  });

  it('scalar and 1d', () => {
    const res = util.assertAndGetBroadcastedShape([6], []);
    expect(res).toEqual([6]);
  });

  it('scalar and 2d', () => {
    const res = util.assertAndGetBroadcastedShape([2, 6], []);
    expect(res).toEqual([2, 6]);
  });

  it('1d and 2d', () => {
    const res = util.assertAndGetBroadcastedShape([6], [2, 6]);
    expect(res).toEqual([2, 6]);
  });

  it('2d and 3d', () => {
    const res = util.assertAndGetBroadcastedShape([2, 6], [7, 2, 6]);
    expect(res).toEqual([7, 2, 6]);
  });

  it('3d and 3d', () => {
    const res = util.assertAndGetBroadcastedShape([1, 1, 6], [7, 2, 6]);
    expect(res).toEqual([7, 2, 6]);
  });

  it('incompatible inner shape', () => {
    const f = () => util.assertAndGetBroadcastedShape([7, 2, 5], [7, 2, 6]);
    expect(f).toThrowError();
  });

  it('incompatible middle shape', () => {
    const f = () => util.assertAndGetBroadcastedShape([7, 3, 6], [7, 2, 6]);
    expect(f).toThrowError();
  });

  it('incompatible due to stricter broadcasting support', () => {
    const f = () => util.assertAndGetBroadcastedShape([7, 3, 6], [7, 1, 6]);
    expect(f).toThrowError();
  });

  it('incompatible due to stricter broadcasting support', () => {
    const f = () => util.assertAndGetBroadcastedShape([7, 1, 1], [7, 1]);
    expect(f).toThrowError();
  });

  it('compatible with stricter broadcasting support', () => {
    const res = util.assertAndGetBroadcastedShape([7, 1, 1], [7, 1, 1]);
    expect(res).toEqual([7, 1, 1]);
  });
});
