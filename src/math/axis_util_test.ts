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

import {ENV} from '../environment';
import * as test_util from '../test_util';
import * as axis_util from './axis_util';
import {Array4D} from './ndarray';

describe('axis_util combineLocations', () => {
  it('rank 4, reduce last 2 dims', () => {
    const loc = axis_util.combineLocations([4, 1], [3, 7], [2, 3]);
    expect(loc).toEqual([4, 1, 3, 7]);
  });

  it('rank 4, reduce first two dims', () => {
    const loc = axis_util.combineLocations([4, 1], [3, 7], [0, 1]);
    expect(loc).toEqual([3, 7, 4, 1]);
  });

  it('rank 4, reduce 1st and 3rd dims', () => {
    const loc = axis_util.combineLocations([4, 1], [3, 7], [0, 2]);
    expect(loc).toEqual([3, 4, 7, 1]);
  });

  it('rank 4, reduce 1st and 4th dims', () => {
    const loc = axis_util.combineLocations([4, 1], [3, 7], [0, 3]);
    expect(loc).toEqual([3, 4, 1, 7]);
  });

  it('rank 3, reduce all dims', () => {
    const loc = axis_util.combineLocations([], [3, 7, 1], [0, 1, 2]);
    expect(loc).toEqual([3, 7, 1]);
  });

  it('rank 2, reduce last dim', () => {
    const loc = axis_util.combineLocations([3], [5], [1]);
    expect(loc).toEqual([3, 5]);
  });

  it('rank 2, reduce first dim', () => {
    const loc = axis_util.combineLocations([3], [5], [0]);
    expect(loc).toEqual([5, 3]);
  });
});

describe('axis_util computeOutAndReduceShapes', () => {
  it('rank 4, reduce all dims', () => {
    const [out, red] =
        axis_util.computeOutAndReduceShapes([3, 7, 2, 4], [0, 1, 2, 3]);
    expect(out).toEqual([]);
    expect(red).toEqual([3, 7, 2, 4]);
  });

  it('rank 4, reduce last 2 dims', () => {
    const [out, red] =
        axis_util.computeOutAndReduceShapes([3, 7, 2, 4], [2, 3]);
    expect(out).toEqual([3, 7]);
    expect(red).toEqual([2, 4]);
  });

  it('rank 4, reduce first 2 dims', () => {
    const [out, red] =
        axis_util.computeOutAndReduceShapes([3, 7, 2, 4], [0, 1]);
    expect(out).toEqual([2, 4]);
    expect(red).toEqual([3, 7]);
  });

  it('rank 4, reduce last 3 dims', () => {
    const [out, red] =
        axis_util.computeOutAndReduceShapes([3, 7, 2, 4], [1, 2, 3]);
    expect(out).toEqual([3]);
    expect(red).toEqual([7, 2, 4]);
  });

  it('rank 4, reduce 1st and 3rd dims', () => {
    const [out, red] =
        axis_util.computeOutAndReduceShapes([3, 7, 2, 4], [0, 2]);
    expect(out).toEqual([7, 4]);
    expect(red).toEqual([3, 2]);
  });

  it('rank 3, reduce all dims', () => {
    const [out, red] =
        axis_util.computeOutAndReduceShapes([3, 7, 2], [0, 1, 2]);
    expect(out).toEqual([]);
    expect(red).toEqual([3, 7, 2]);
  });
});

describe('axis_util axesAreInnerMostDims', () => {
  it('rank 4, reduce last dim', () => {
    const res = axis_util.axesAreInnerMostDims([3], 4);
    expect(res).toBe(true);
  });

  it('rank 4, reduce last 2 dims', () => {
    const res = axis_util.axesAreInnerMostDims([2, 3], 4);
    expect(res).toBe(true);
  });

  it('rank 4, reduce last 3 dims', () => {
    const res = axis_util.axesAreInnerMostDims([1, 2, 3], 4);
    expect(res).toBe(true);
  });

  it('rank 4, reduce all dims', () => {
    const res = axis_util.axesAreInnerMostDims([0, 1, 2, 3], 4);
    expect(res).toBe(true);
  });

  it('rank 4, reduce all but 2nd', () => {
    const res = axis_util.axesAreInnerMostDims([0, 2, 3], 4);
    expect(res).toBe(false);
  });

  it('rank 4, reduce all but 3rd', () => {
    const res = axis_util.axesAreInnerMostDims([0, 1, 3], 4);
    expect(res).toBe(false);
  });

  it('rank 4, reduce all but last', () => {
    const res = axis_util.axesAreInnerMostDims([0, 1, 2], 4);
    expect(res).toBe(false);
  });
});

describe('axis_util expandShapeToKeepDim', () => {
  it('2d -> 1d axis=0', () => {
    const shape = axis_util.expandShapeToKeepDim([2], [0]);
    expect(shape).toEqual([1, 2]);
  });

  it('2d -> 1d axis=1', () => {
    const shape = axis_util.expandShapeToKeepDim([4], [1]);
    expect(shape).toEqual([4, 1]);
  });

  it('3d -> 1d axis=1,2', () => {
    const shape = axis_util.expandShapeToKeepDim([7], [1, 2]);
    expect(shape).toEqual([7, 1, 1]);
  });

  it('3d -> 2d axis=1', () => {
    const shape = axis_util.expandShapeToKeepDim([7, 3], [1]);
    expect(shape).toEqual([7, 1, 3]);
  });
});

describe('axis_util getPermAxes', () => {
  it('all axes, no perm is needed', () => {
    const perm = axis_util.getAxesPermutation([0, 1, 2], 3);
    expect(perm).toBeNull();
  });

  it('no axes, no perm is needed', () => {
    const perm = axis_util.getAxesPermutation([], 3);
    expect(perm).toBeNull();
  });

  it('inner most 2 axes, no perm is needed', () => {
    const perm = axis_util.getAxesPermutation([2, 3], 4);
    expect(perm).toBeNull();
  });

  it('outer most axis, perm is needed', () => {
    const perm = axis_util.getAxesPermutation([0], 4);
    expect(perm).toEqual([1, 2, 3, 0]);
  });

  it('2 outer most axes, perm is needed', () => {
    const perm = axis_util.getAxesPermutation([0, 1], 4);
    expect(perm).toEqual([2, 3, 0, 1]);
  });
});

describe('axis_util parseAxisParam', () => {
  it('axis=null returns no axes for scalar', () => {
    const axis: number = null;
    const shape: number[] = [];
    expect(axis_util.parseAxisParam(axis, shape)).toEqual([]);
  });

  it('axis=null returns 0 axis for array1d', () => {
    const axis: number = null;
    const shape = [4];
    expect(axis_util.parseAxisParam(axis, shape)).toEqual([0]);
  });

  it('axis=null returns all axes for array3d', () => {
    const axis: number[] = null;
    const shape = [3, 1, 2];
    expect(axis_util.parseAxisParam(axis, shape)).toEqual([0, 1, 2]);
  });

  it('axis as a single number', () => {
    const axis = 1;
    const shape = [3, 1, 2];
    expect(axis_util.parseAxisParam(axis, shape)).toEqual([1]);
  });

  it('axis as single negative number', () => {
    const axis = -1;
    const shape = [3, 1, 2];
    expect(axis_util.parseAxisParam(axis, shape)).toEqual([2]);

    const axis2 = -2;
    expect(axis_util.parseAxisParam(axis2, shape)).toEqual([1]);

    const axis3 = -3;
    expect(axis_util.parseAxisParam(axis3, shape)).toEqual([0]);
  });

  it('axis as list of negative numbers', () => {
    const axis = [-1, -3];
    const shape = [3, 1, 2];
    expect(axis_util.parseAxisParam(axis, shape)).toEqual([2, 0]);
  });

  it('axis as list of positive numbers', () => {
    const axis = [0, 2];
    const shape = [3, 1, 2];
    expect(axis_util.parseAxisParam(axis, shape)).toEqual([0, 2]);
  });

  it('axis as combo of positive and negative numbers', () => {
    const axis = [0, -1];
    const shape = [3, 1, 2];
    expect(axis_util.parseAxisParam(axis, shape)).toEqual([0, 2]);
  });

  it('axis out of range throws error', () => {
    const axis = -4;
    const shape = [3, 1, 2];
    expect(() => axis_util.parseAxisParam(axis, shape)).toThrowError();

    const axis2 = 4;
    expect(() => axis_util.parseAxisParam(axis2, shape)).toThrowError();
  });

  it('axis a list with one number out of range throws error', () => {
    const axis = [0, 4];
    const shape = [3, 1, 2];
    expect(() => axis_util.parseAxisParam(axis, shape)).toThrowError();
  });

  it('axis with decimal value throws error', () => {
    const axis = 0.5;
    const shape = [3, 1, 2];
    expect(() => axis_util.parseAxisParam(axis, shape)).toThrowError();
  });
});

describe('axis_util getUndoAxesPermutation', () => {
  it('4d axes', () => {
    const axes = [2, 0, 1, 3];
    expect(axis_util.getUndoAxesPermutation(axes)).toEqual([1, 2, 0, 3]);
  });

  it('3d axes, no perm', () => {
    const axes = [0, 1, 2];
    expect(axis_util.getUndoAxesPermutation(axes)).toEqual([0, 1, 2]);
  });

  it('3d axes, complete flip', () => {
    const axes = [2, 1, 0];
    expect(axis_util.getUndoAxesPermutation(axes)).toEqual([2, 1, 0]);
  });

  it('4d array with values', () => {
    const axes = [2, 0, 1, 3];
    const undoPermutation = axis_util.getUndoAxesPermutation(axes);

    const a = Array4D.randNormal([2, 3, 4, 5]);
    const aT = ENV.math.transpose(a, axes);
    const aTT = ENV.math.transpose(aT, undoPermutation);
    test_util.expectArraysClose(a, aTT);
  });
});
