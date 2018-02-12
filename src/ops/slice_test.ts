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

import * as dl from '../index';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, describeWithFlags, expectArraysClose, expectNumbersClose} from '../test_util';
import {Rank} from '../types';

describeWithFlags('slice1d', ALL_ENVS, () => {
  it('slices 1x1 into 1x1 (effectively a copy)', () => {
    const a = dl.tensor1d([5]);
    const result = dl.slice1d(a, 0, 1);

    expect(result.shape).toEqual([1]);
    expectNumbersClose(result.get(0), 5);
  });

  it('slices 5x1 into shape 2x1 starting at 3', () => {
    const a = dl.tensor1d([1, 2, 3, 4, 5]);
    const result = dl.slice1d(a, 3, 2);

    expect(result.shape).toEqual([2]);
    expectArraysClose(result, [4, 5]);
  });

  it('slices 5x1 into shape 3x1 starting at 1', () => {
    const a = dl.tensor1d([1, 2, 3, 4, 5]);
    const result = dl.slice1d(a, 1, 3);

    expect(result.shape).toEqual([3]);
    expectArraysClose(result, [2, 3, 4]);
  });
});

describeWithFlags('slice2d', ALL_ENVS, () => {
  it('slicing a 1x1 from a 1x1 returns a 1x1', () => {
    const a = dl.tensor2d([0], [1, 1]);
    const b = dl.slice2d(a, [0, 0], [1, 1]);
    expect(b.shape).toEqual([1, 1]);
  });

  it('returns a tensor of slice size', () => {
    const a = dl.zeros<Rank.R2>([100, 100]);
    const b = dl.slice2d(a, [0, 0], [12, 34]);
    expect(b.shape).toEqual([12, 34]);
  });

  it('returns the upper-left submatrix when begin is [0, 0]', () => {
    const a = dl.randomUniform<Rank.R2>([10, 10], -1, 1);
    const b = dl.slice2d(a, [0, 0], [2, 2]);
    const aValues = a.dataSync();

    expectArraysClose(b, [aValues[0], aValues[1], aValues[10], aValues[11]]);
  });

  it('returns the rectangle specified', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3]);
    const b = dl.slice2d(a, [1, 1], [3, 2]);

    expectArraysClose(b, [5, 6, 8, 9, 11, 12]);
  });

  it('throws when requesting out of bounds slice', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3]);
    expect(() => dl.slice2d(a, [1, 1], [10, 10])).toThrowError();
  });
});

describeWithFlags('slice3d', ALL_ENVS, () => {
  it('slices 1x1x1 into shape 1x1x1 (effectively a copy)', () => {
    const a = dl.tensor3d([[[5]]], [1, 1, 1]);
    const result = a.slice([0, 0, 0], [1, 1, 1]);

    expect(result.shape).toEqual([1, 1, 1]);
    expectArraysClose(result, [5]);
  });

  it('slices 2x2x2 array into 1x2x2 starting at [1, 0, 0]', () => {
    const a = dl.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const result = a.slice([1, 0, 0], [1, 2, 2]);

    expect(result.shape).toEqual([1, 2, 2]);
    expectArraysClose(result, [5, 6, 7, 8]);
  });

  it('slices 2x2x2 array into 2x1x1 starting at [0, 1, 1]', () => {
    const a = dl.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const result = a.slice([0, 1, 1], [2, 1, 1]);

    expect(result.shape).toEqual([2, 1, 1]);
    expectArraysClose(result, [4, 8]);
  });
});

describeWithFlags('slice4d', ALL_ENVS, () => {
  it('slices 1x1x1x1 into shape 1x1x1x1 (effectively a copy)', () => {
    const a = dl.tensor4d([[[[5]]]], [1, 1, 1, 1]);
    const result = a.slice([0, 0, 0, 0], [1, 1, 1, 1]);

    expect(result.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(result, [5]);
  });

  it('slices 2x2x2x2 array into 1x2x2x2 starting at [1, 0, 0, 0]', () => {
    const a = dl.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88],
        [2, 2, 2, 2],
    );
    const result = a.slice([1, 0, 0, 0], [1, 2, 2, 2]);

    expect(result.shape).toEqual([1, 2, 2, 2]);
    expectArraysClose(result, [11, 22, 33, 44, 55, 66, 77, 88]);
  });

  it('slices 2x2x2x2 array into 2x1x1x1 starting at [0, 1, 1, 1]', () => {
    const a = dl.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88], [2, 2, 2, 2]);
    const result = a.slice([0, 1, 1, 1], [2, 1, 1, 1]);

    expect(result.shape).toEqual([2, 1, 1, 1]);
    expectArraysClose(result, [8, 88]);
  });
});
