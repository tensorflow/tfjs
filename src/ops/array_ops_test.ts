/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import {ALL_ENVS, describeWithFlags, expectArraysClose, expectArraysEqual, expectValuesInRange} from '../test_util';
import * as util from '../util';
import {expectArrayInMeanStdRange, jarqueBeraNormalityTest} from './rand_util';

describeWithFlags('zeros', ALL_ENVS, () => {
  it('1D default dtype', () => {
    const a: dl.Tensor1D = dl.zeros([3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(a, [0, 0, 0]);
  });

  it('1D float32 dtype', () => {
    const a = dl.zeros([3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(a, [0, 0, 0]);
  });

  it('1D int32 dtype', () => {
    const a = dl.zeros([3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(a, [0, 0, 0]);
  });

  it('1D bool dtype', () => {
    const a = dl.zeros([3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(a, [0, 0, 0]);
  });

  it('2D default dtype', () => {
    const a = dl.zeros([3, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('2D float32 dtype', () => {
    const a = dl.zeros([3, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('2D int32 dtype', () => {
    const a = dl.zeros([3, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });

  it('2D bool dtype', () => {
    const a = dl.zeros([3, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2]);
    expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });

  it('3D default dtype', () => {
    const a = dl.zeros([2, 2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysClose(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D float32 dtype', () => {
    const a = dl.zeros([2, 2, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysClose(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D int32 dtype', () => {
    const a = dl.zeros([2, 2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysEqual(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D bool dtype', () => {
    const a = dl.zeros([2, 2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysEqual(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('4D default dtype', () => {
    const a = dl.zeros([3, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('4D float32 dtype', () => {
    const a = dl.zeros([3, 2, 1, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('4D int32 dtype', () => {
    const a = dl.zeros([3, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });

  it('4D bool dtype', () => {
    const a = dl.zeros([3, 2, 1, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });
});

describeWithFlags('ones', ALL_ENVS, () => {
  it('1D default dtype', () => {
    const a = dl.ones([3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(a, [1, 1, 1]);
  });

  it('1D float32 dtype', () => {
    const a = dl.ones([3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(a, [1, 1, 1]);
  });

  it('1D int32 dtype', () => {
    const a = dl.ones([3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(a, [1, 1, 1]);
  });

  it('1D bool dtype', () => {
    const a = dl.ones([3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(a, [1, 1, 1]);
  });

  it('2D default dtype', () => {
    const a = dl.ones([3, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('2D float32 dtype', () => {
    const a = dl.ones([3, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('2D int32 dtype', () => {
    const a = dl.ones([3, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });

  it('2D bool dtype', () => {
    const a = dl.ones([3, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2]);
    expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });

  it('3D default dtype', () => {
    const a = dl.ones([2, 2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysClose(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('3D float32 dtype', () => {
    const a = dl.ones([2, 2, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysClose(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('3D int32 dtype', () => {
    const a = dl.ones([2, 2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysEqual(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('3D bool dtype', () => {
    const a = dl.ones([2, 2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysEqual(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('4D default dtype', () => {
    const a = dl.ones([3, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('4D float32 dtype', () => {
    const a = dl.ones([3, 2, 1, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('4D int32 dtype', () => {
    const a = dl.ones([3, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });

  it('4D bool dtype', () => {
    const a = dl.ones([3, 2, 1, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });
});

describeWithFlags('zerosLike', ALL_ENVS, () => {
  it('1D default dtype', () => {
    const a = dl.tensor1d([1, 2, 3]);
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(b, [0, 0, 0]);
  });

  it('1D float32 dtype', () => {
    const a = dl.tensor1d([1, 2, 3], 'float32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(b, [0, 0, 0]);
  });

  it('1D int32 dtype', () => {
    const a = dl.tensor1d([1, 2, 3], 'int32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(b, [0, 0, 0]);
  });

  it('1D bool dtype', () => {
    const a = dl.tensor1d([1, 2, 3], 'bool');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(b, [0, 0, 0]);
  });

  it('2D default dtype', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('2D float32 dtype', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('2D int32 dtype', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('2D bool dtype', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'bool');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('3D default dtype', () => {
    const a = dl.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('3D float32 dtype', () => {
    const a = dl.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('3D int32 dtype', () => {
    const a = dl.tensor3d([1, 2, 3, 4], [2, 2, 1], 'int32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('3D bool dtype', () => {
    const a = dl.tensor3d([1, 2, 3, 4], [2, 2, 1], 'bool');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('4D default dtype', () => {
    const a = dl.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('4D float32 dtype', () => {
    const a = dl.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('4D int32 dtype', () => {
    const a = dl.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'int32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('4D bool dtype', () => {
    const a = dl.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'bool');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });
});

describeWithFlags('onesLike', ALL_ENVS, () => {
  it('1D default dtype', () => {
    const a = dl.tensor1d([1, 2, 3]);
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(b, [1, 1, 1]);
  });

  it('1D float32 dtype', () => {
    const a = dl.tensor1d([1, 2, 3], 'float32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(b, [1, 1, 1]);
  });

  it('1D int32 dtype', () => {
    const a = dl.tensor1d([1, 2, 3], 'int32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(b, [1, 1, 1]);
  });

  it('1D bool dtype', () => {
    const a = dl.tensor1d([1, 2, 3], 'bool');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(b, [1, 1, 1]);
  });

  it('2D default dtype', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('2D float32 dtype', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('2D int32 dtype', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('2D bool dtype', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'bool');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('3D default dtype', () => {
    const a = dl.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('3D float32 dtype', () => {
    const a = dl.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('3D int32 dtype', () => {
    const a = dl.tensor3d([1, 2, 3, 4], [2, 2, 1], 'int32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('3D bool dtype', () => {
    const a = dl.tensor3d([1, 2, 3, 4], [2, 2, 1], 'bool');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('4D default dtype', () => {
    const a = dl.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('4D float32 dtype', () => {
    const a = dl.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('4D int32 dtype', () => {
    const a = dl.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'int32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('4D bool dtype', () => {
    const a = dl.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'bool');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });
});

describeWithFlags('rand', ALL_ENVS, () => {
  it('should return a random 1D float32 array', () => {
    const shape: [number] = [10];

    // Enusre defaults to float32 w/o type:
    let result = dl.rand(shape, () => util.randUniform(0, 2));
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2);

    result = dl.rand(shape, () => util.randUniform(0, 1.5));
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 1D int32 array', () => {
    const shape: [number] = [10];
    const result = dl.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 1D bool array', () => {
    const shape: [number] = [10];
    const result = dl.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 2D float32 array', () => {
    const shape = [3, 4];

    // Enusre defaults to float32 w/o type:
    let result = dl.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = dl.rand(shape, () => util.randUniform(0, 1.5), 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 2D int32 array', () => {
    const shape = [3, 4];
    const result = dl.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 2D bool array', () => {
    const shape = [3, 4];
    const result = dl.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 3D float32 array', () => {
    const shape = [3, 4, 5];

    // Enusre defaults to float32 w/o type:
    let result = dl.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = dl.rand(shape, () => util.randUniform(0, 1.5), 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 3D int32 array', () => {
    const shape = [3, 4, 5];
    const result = dl.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 3D bool array', () => {
    const shape = [3, 4, 5];
    const result = dl.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 4D float32 array', () => {
    const shape = [3, 4, 5, 6];

    // Enusre defaults to float32 w/o type:
    let result = dl.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = dl.rand(shape, () => util.randUniform(0, 1.5));
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 4D int32 array', () => {
    const shape = [3, 4, 5, 6];
    const result = dl.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 4D bool array', () => {
    const shape = [3, 4, 5, 6];
    const result = dl.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });
});

describeWithFlags('randomNormal', ALL_ENVS, () => {
  const SEED = 2002;
  const EPSILON = 0.05;

  it('should return a float32 1D of random normal values', () => {
    const SAMPLES = 10000;

    // Ensure defaults to float32.
    let result = dl.randomNormal([SAMPLES], 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);

    result = dl.randomNormal([SAMPLES], 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
  });

  it('should return a int32 1D of random normal values', () => {
    const SAMPLES = 10000;
    const result = dl.randomNormal([SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a float32 2D of random normal values', () => {
    const SAMPLES = 250;

    // Ensure defaults to float32.
    let result = dl.randomNormal([SAMPLES, SAMPLES], 0, 2.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 2.5, EPSILON);

    result = dl.randomNormal([SAMPLES, SAMPLES], 0, 3.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
  });

  it('should return a int32 2D of random normal values', () => {
    const SAMPLES = 100;
    const result = dl.randomNormal([SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a float32 3D of random normal values', () => {
    const SAMPLES = 50;

    // Ensure defaults to float32.
    let result =
        dl.randomNormal([SAMPLES, SAMPLES, SAMPLES], 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);

    result =
        dl.randomNormal([SAMPLES, SAMPLES, SAMPLES], 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
  });

  it('should return a int32 3D of random normal values', () => {
    const SAMPLES = 50;
    const result =
        dl.randomNormal([SAMPLES, SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a float32 4D of random normal values', () => {
    const SAMPLES = 25;

    // Ensure defaults to float32.
    let result = dl.randomNormal(
        [SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);

    result = dl.randomNormal(
        [SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
  });

  it('should return a int32 4D of random normal values', () => {
    const SAMPLES = 25;

    const result = dl.randomNormal(
        [SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });
});

describeWithFlags('truncatedNormal', ALL_ENVS, () => {
  // Expect slightly higher variances for truncated values.
  const EPSILON = 0.60;
  const SEED = 2002;

  function assertTruncatedValues(array: dl.Tensor, mean: number, stdv: number) {
    const bounds = mean + stdv * 2;
    const values = array.dataSync();
    for (let i = 0; i < values.length; i++) {
      expect(Math.abs(values[i])).toBeLessThanOrEqual(bounds);
    }
  }

  it('should return a random 1D float32 array', () => {
    const shape: [number] = [1000];

    // Ensure defaults to float32 w/o type:
    let result = dl.truncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = dl.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a randon 1D int32 array', () => {
    const shape: [number] = [1000];
    const result = dl.truncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });

  it('should return a 2D float32 array', () => {
    const shape: [number, number] = [50, 50];

    // Ensure defaults to float32 w/o type:
    let result = dl.truncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = dl.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a 2D int32 array', () => {
    const shape: [number, number] = [50, 50];
    const result = dl.truncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });

  it('should return a 3D float32 array', () => {
    const shape: [number, number, number] = [10, 10, 10];

    // Ensure defaults to float32 w/o type:
    let result = dl.truncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = dl.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a 3D int32 array', () => {
    const shape: [number, number, number] = [10, 10, 10];
    const result = dl.truncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });

  it('should return a 4D float32 array', () => {
    const shape: [number, number, number, number] = [5, 5, 5, 5];

    // Ensure defaults to float32 w/o type:
    let result = dl.truncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = dl.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a 4D int32 array', () => {
    const shape: [number, number, number, number] = [5, 5, 5, 5];
    const result = dl.truncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });
});

describeWithFlags('randomUniform', ALL_ENVS, () => {
  it('should return a random 1D float32 array', () => {
    const shape: [number] = [10];

    // Enusre defaults to float32 w/o type:
    let result = dl.randomUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = dl.randomUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 1D int32 array', () => {
    const shape: [number] = [10];
    const result = dl.randomUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 1D bool array', () => {
    const shape: [number] = [10];
    const result = dl.randomUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 2D float32 array', () => {
    const shape: [number, number] = [3, 4];

    // Enusre defaults to float32 w/o type:
    let result = dl.randomUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = dl.randomUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 2D int32 array', () => {
    const shape: [number, number] = [3, 4];
    const result = dl.randomUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 2D bool array', () => {
    const shape: [number, number] = [3, 4];
    const result = dl.randomUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 3D float32 array', () => {
    const shape: [number, number, number] = [3, 4, 5];

    // Enusre defaults to float32 w/o type:
    let result = dl.randomUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = dl.randomUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 3D int32 array', () => {
    const shape: [number, number, number] = [3, 4, 5];
    const result = dl.randomUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 3D bool array', () => {
    const shape: [number, number, number] = [3, 4, 5];
    const result = dl.randomUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 4D float32 array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];

    // Enusre defaults to float32 w/o type:
    let result = dl.randomUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = dl.randomUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 4D int32 array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];
    const result = dl.randomUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 4D bool array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];
    const result = dl.randomUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });
});

describeWithFlags('fromPixels', ALL_ENVS, () => {
  it('ImageData 1x1x3', () => {
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 0;
    pixels.data[1] = 80;
    pixels.data[2] = 160;
    pixels.data[3] = 240;

    const array = dl.fromPixels(pixels, 3);

    expectArraysEqual(array, [0, 80, 160]);
  });

  it('ImageData 1x1x4', () => {
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 0;
    pixels.data[1] = 80;
    pixels.data[2] = 160;
    pixels.data[3] = 240;

    const array = dl.fromPixels(pixels, 4);

    expectArraysEqual(array, [0, 80, 160, 240]);
  });

  it('ImageData 2x2x3', () => {
    const pixels = new ImageData(2, 2);

    for (let i = 0; i < 8; i++) {
      pixels.data[i] = i * 2;
    }
    for (let i = 8; i < 16; i++) {
      pixels.data[i] = i * 2;
    }

    const array = dl.fromPixels(pixels, 3);

    expectArraysEqual(array, [0, 2, 4, 8, 10, 12, 16, 18, 20, 24, 26, 28]);
  });

  it('ImageData 2x2x4', () => {
    const pixels = new ImageData(2, 2);
    for (let i = 0; i < 8; i++) {
      pixels.data[i] = i * 2;
    }
    for (let i = 8; i < 16; i++) {
      pixels.data[i] = i * 2;
    }

    const array = dl.fromPixels(pixels, 4);

    expectArraysClose(
        array,
        new Int32Array(
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]));
  });

  it('fromPixels, 3 channels', () => {
    const pixels = new ImageData(1, 2);
    pixels.data[0] = 2;
    pixels.data[1] = 3;
    pixels.data[2] = 4;
    pixels.data[3] = 255;  // Not used.
    pixels.data[4] = 5;
    pixels.data[5] = 6;
    pixels.data[6] = 7;
    pixels.data[7] = 255;  // Not used.
    const res = dl.fromPixels(pixels, 3);
    expect(res.shape).toEqual([2, 1, 3]);
    expect(res.dtype).toBe('int32');
    expectArraysClose(res, [2, 3, 4, 5, 6, 7]);
  });

  it('fromPixels, reshape, then do dl.add()', () => {
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 2;
    pixels.data[1] = 3;
    pixels.data[2] = 4;
    pixels.data[3] = 255;  // Not used.
    const a = dl.fromPixels(pixels, 3).reshape([1, 1, 1, 3]);
    const res = a.add(dl.scalar(2, 'int32'));
    expect(res.shape).toEqual([1, 1, 1, 3]);
    expect(res.dtype).toBe('int32');
    expectArraysClose(res, [4, 5, 6]);
  });

  it('fromPixels + fromPixels', () => {
    const pixelsA = new ImageData(1, 1);
    pixelsA.data[0] = 255;
    pixelsA.data[1] = 3;
    pixelsA.data[2] = 4;
    pixelsA.data[3] = 255;  // Not used.
    const pixelsB = new ImageData(1, 1);
    pixelsB.data[0] = 5;
    pixelsB.data[1] = 6;
    pixelsB.data[2] = 7;
    pixelsB.data[3] = 255;  // Not used.
    const a = dl.fromPixels(pixelsA, 3).toFloat();
    const b = dl.fromPixels(pixelsB, 3).toFloat();
    const res = a.add(b);
    expect(res.shape).toEqual([1, 1, 3]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(res, [260, 9, 11]);
  });
});

describeWithFlags('clone', ALL_ENVS, () => {
  it('1D default dtype', () => {
    const a = dl.tensor1d([1, 2, 3]);
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(b, [1, 2, 3]);
  });

  it('1D float32 dtype', () => {
    const a = dl.tensor1d([1, 2, 3], 'float32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(b, [1, 2, 3]);
  });

  it('1D int32 dtype', () => {
    const a = dl.tensor1d([1, 2, 3], 'int32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(b, [1, 2, 3]);
  });

  it('1D bool dtype', () => {
    const a = dl.tensor1d([1, 2, 3], 'bool');
    const b = dl.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(b, [1, 1, 1]);
  });

  it('2D default dtype', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('2D float32 dtype', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('2D int32 dtype', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(b, [1, 2, 3, 4]);
  });

  it('2D bool dtype', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'bool');
    const b = dl.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('3D default dtype', () => {
    const a = dl.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('3D float32 dtype', () => {
    const a = dl.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('3D int32 dtype', () => {
    const a = dl.tensor3d([1, 2, 3, 4], [2, 2, 1], 'int32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(b, [1, 2, 3, 4]);
  });

  it('3D bool dtype', () => {
    const a = dl.tensor3d([1, 2, 3, 4], [2, 2, 1], 'bool');
    const b = dl.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('4D default dtype', () => {
    const a = dl.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('4D float32 dtype', () => {
    const a = dl.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('4D int32 dtype', () => {
    const a = dl.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'int32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(b, [1, 2, 3, 4]);
  });

  it('4D bool dtype', () => {
    const a = dl.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'bool');
    const b = dl.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });
});

describeWithFlags('tile', ALL_ENVS, () => {
  it('1D (tile)', () => {
    const t = dl.tensor1d([1, 2, 3]);
    const t2 = dl.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expectArraysClose(t2, [1, 2, 3, 1, 2, 3]);
  });

  it('2D (tile)', () => {
    const t = dl.tensor2d([1, 11, 2, 22], [2, 2]);
    let t2 = dl.tile(t, [1, 2]);

    expect(t2.shape).toEqual([2, 4]);
    expectArraysClose(t2, [1, 11, 1, 11, 2, 22, 2, 22]);

    t2 = dl.tile(t, [2, 1]);
    expect(t2.shape).toEqual([4, 2]);
    expectArraysClose(t2, [1, 11, 2, 22, 1, 11, 2, 22]);

    t2 = dl.tile(t, [2, 2]);
    expect(t2.shape).toEqual([4, 4]);
    expectArraysClose(
        t2, [1, 11, 1, 11, 2, 22, 2, 22, 1, 11, 1, 11, 2, 22, 2, 22]);
  });

  it('3D (tile)', () => {
    const t = dl.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const t2 = dl.tile(t, [1, 2, 1]);

    expect(t2.shape).toEqual([2, 4, 2]);
    expectArraysClose(t2, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
  });

  it('propagates NaNs', () => {
    const t = dl.tensor1d([1, 2, NaN]);

    const t2 = dl.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expectArraysClose(t2, [1, 2, NaN, 1, 2, NaN]);
  });

  it('1D bool (tile)', () => {
    const t = dl.tensor1d([true, false, true], 'bool');
    const t2 = dl.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(t2, [1, 0, 1, 1, 0, 1]);
  });

  it('2D bool (tile)', () => {
    const t = dl.tensor2d([true, false, true, true], [2, 2], 'bool');
    let t2 = dl.tile(t, [1, 2]);

    expect(t2.shape).toEqual([2, 4]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(t2, [1, 0, 1, 0, 1, 1, 1, 1]);

    t2 = dl.tile(t, [2, 1]);
    expect(t2.shape).toEqual([4, 2]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(t2, [1, 0, 1, 1, 1, 0, 1, 1]);

    t2 = dl.tile(t, [2, 2]);
    expect(t2.shape).toEqual([4, 4]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(t2, [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]);
  });

  it('3D bool (tile)', () => {
    const t = dl.tensor3d(
        [true, false, true, false, true, false, true, false], [2, 2, 2],
        'bool');
    const t2 = dl.tile(t, [1, 2, 1]);

    expect(t2.shape).toEqual([2, 4, 2]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(t2, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
  });

  it('bool propagates NaNs', () => {
    const t = dl.tensor1d([true, false, NaN] as boolean[], 'bool');
    const t2 = dl.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(
        t2, [1, 0, util.getNaN('bool'), 1, 0, util.getNaN('bool')]);
  });

  it('1D int32 (tile)', () => {
    const t = dl.tensor1d([1, 2, 5], 'int32');
    const t2 = dl.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(t2, [1, 2, 5, 1, 2, 5]);
  });

  it('2D int32 (tile)', () => {
    const t = dl.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    let t2 = dl.tile(t, [1, 2]);

    expect(t2.shape).toEqual([2, 4]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(t2, [1, 2, 1, 2, 3, 4, 3, 4]);

    t2 = dl.tile(t, [2, 1]);
    expect(t2.shape).toEqual([4, 2]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(t2, [1, 2, 3, 4, 1, 2, 3, 4]);

    t2 = dl.tile(t, [2, 2]);
    expect(t2.shape).toEqual([4, 4]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(t2, [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4]);
  });

  it('3D int32 (tile)', () => {
    const t = dl.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2], 'int32');
    const t2 = dl.tile(t, [1, 2, 1]);

    expect(t2.shape).toEqual([2, 4, 2]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(t2, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
  });

  it('int32 propagates NaNs', () => {
    const t = dl.tensor1d([1, 3, NaN], 'int32');
    const t2 = dl.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(
        t2, [1, 3, util.getNaN('int32'), 1, 3, util.getNaN('int32')]);
  });

  it('1D (tile) gradient', () => {
    const x = dl.tensor1d([1, 2, 3]);
    const dy = dl.tensor1d([0.1, 0.2, 0.3, 1, 2, 3, 10, 20, 30]);
    const gradients = dl.grad(x => dl.tile(x, [3]))(x, dy);
    expectArraysClose(gradients, dl.tensor1d([11.1, 22.2, 33.3]));
  });

  it('2D (tile) gradient', () => {
    const x = dl.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const dy = dl.tensor2d([[1, 2, 10, 20], [3, 4, 30, 40]], [2, 4]);
    const gradients = dl.grad(x => dl.tile(x, [1, 2]))(x, dy);
    expectArraysClose(gradients, dl.tensor2d([[11, 22], [33, 44]], [2, 2]));
  });

  it('3D (tile) gradient', () => {
    const x = dl.tensor3d([[[1], [2]], [[3], [4]]], [2, 2, 1]);
    const dy = dl.tensor3d([[[1, 10], [2, 20]], [[3, 30], [4, 40]]], [2, 2, 2]);
    const gradients = dl.grad(x => dl.tile(x, [1, 1, 2]))(x, dy);
    expectArraysClose(
        gradients, dl.tensor3d([[[11], [22]], [[33], [44]]], [2, 2, 1]));
  });

  it('4D (tile) gradient', () => {
    const x = dl.tensor4d([[[[1]], [[2]]], [[[3]], [[4]]]], [2, 2, 1, 1]);
    const dy = dl.tensor4d(
        [
          [[[1, 10], [100, 1000]], [[2, 20], [200, 2000]]],
          [[[3, 30], [300, 3000]], [[4, 40], [400, 4000]]]
        ],
        [2, 2, 2, 2]);
    const gradients = dl.grad(x => dl.tile(x, [1, 1, 2, 2]))(x, dy);
    expectArraysClose(
        gradients,
        dl.tensor4d(
            [[[[1111]], [[2222]]], [[[3333]], [[4444]]]], [2, 2, 1, 1]));
  });
});

describeWithFlags('gather', ALL_ENVS, () => {
  it('1D (gather)', () => {
    const t = dl.tensor1d([1, 2, 3]);

    const t2 = dl.gather(t, dl.tensor1d([0, 2, 0, 1], 'int32'), 0);

    expect(t2.shape).toEqual([4]);
    expectArraysClose(t2, [1, 3, 1, 2]);
  });

  it('2D (gather)', () => {
    const t = dl.tensor2d([1, 11, 2, 22], [2, 2]);
    let t2 = dl.gather(t, dl.tensor1d([1, 0, 0, 1], 'int32'), 0);
    expect(t2.shape).toEqual([4, 2]);
    expectArraysClose(t2, [2, 22, 1, 11, 1, 11, 2, 22]);

    t2 = dl.gather(t, dl.tensor1d([1, 0, 0, 1], 'int32'), 1);
    expect(t2.shape).toEqual([2, 4]);
    expectArraysClose(t2, [11, 1, 1, 11, 22, 2, 2, 22]);
  });

  it('3D (gather)', () => {
    const t = dl.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);

    const t2 = dl.gather(t, dl.tensor1d([1, 0, 0, 1], 'int32'), 2);

    expect(t2.shape).toEqual([2, 2, 4]);
    expectArraysClose(t2, [2, 1, 1, 2, 4, 3, 3, 4, 6, 5, 5, 6, 8, 7, 7, 8]);
  });

  it('bool (gather)', () => {
    const t = dl.tensor1d([true, false, true], 'bool');

    const t2 = dl.gather(t, dl.tensor1d([0, 2, 0, 1], 'int32'), 0);

    expect(t2.shape).toEqual([4]);
    expect(t2.dtype).toBe('bool');
    expect(t2.dataSync()).toEqual(new Uint8Array([1, 1, 1, 0]));
  });

  it('int32 (gather)', () => {
    const t = dl.tensor1d([1, 2, 5], 'int32');

    const t2 = dl.gather(t, dl.tensor1d([0, 2, 0, 1], 'int32'), 0);

    expect(t2.shape).toEqual([4]);
    expect(t2.dtype).toBe('int32');
    expect(t2.dataSync()).toEqual(new Int32Array([1, 5, 1, 2]));
  });

  it('propagates NaNs', () => {
    const t = dl.tensor1d([1, 2, NaN]);

    const t2 = dl.gather(t, dl.tensor1d([0, 2, 0, 1], 'int32'), 0);

    expect(t2.shape).toEqual([4]);
    expectArraysClose(t2, [1, NaN, 1, 2]);
  });
});

describeWithFlags('oneHot', ALL_ENVS, () => {
  it('Depth 1 throws error', () => {
    const indices = dl.tensor1d([0, 0, 0]);
    expect(() => dl.oneHot(indices, 1)).toThrowError();
  });

  it('Depth 2, diagonal', () => {
    const indices = dl.tensor1d([0, 1]);
    const res = dl.oneHot(indices, 2);

    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(res, [1, 0, 0, 1]);
  });

  it('Depth 2, transposed diagonal', () => {
    const indices = dl.tensor1d([1, 0]);
    const res = dl.oneHot(indices, 2);

    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(res, [0, 1, 1, 0]);
  });

  it('Depth 3, 4 events', () => {
    const indices = dl.tensor1d([2, 1, 2, 0]);
    const res = dl.oneHot(indices, 3);

    expect(res.shape).toEqual([4, 3]);
    expectArraysClose(res, [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]);
  });

  it('Depth 2 onValue=3, offValue=-2', () => {
    const indices = dl.tensor1d([0, 1]);
    const res = dl.oneHot(indices, 2, 3, -2);

    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(res, [3, -2, -2, 3]);
  });
});

describeWithFlags('linspace', ALL_ENVS, () => {
  it('start stop', () => {
    const a = dl.linspace(1, 10, 10);
    expectArraysEqual(a, [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    expect(a.shape).toEqual([10]);

    const b = dl.linspace(12, 17, 8);
    expectArraysClose(b, [
      12., 12.71428571, 13.42857143, 14.14285714, 14.85714286, 15.57142857,
      16.28571429, 17.
    ]);
    expect(b.shape).toEqual([8]);

    const c = dl.linspace(9, 0, 6);
    expectArraysClose(c, [9., 7.2, 5.4, 3.6, 1.8, 0.]);
    expect(c.shape).toEqual([6]);
  });

  it('negative start stop', () => {
    const a = dl.linspace(-4, 5, 6);
    expectArraysClose(a, [-4., -2.2, -0.4, 1.4, 3.2, 5.]);
    expect(a.shape).toEqual([6]);
  });

  it('start negative stop', () => {
    const a = dl.linspace(4, -5, 6);
    expectArraysClose(a, [4., 2.2, 0.4, -1.4, -3.2, -5.]);
    expect(a.shape).toEqual([6]);
  });

  it('negative start negative stop', () => {
    const a = dl.linspace(-4, -5, 6);
    expectArraysClose(a, [-4., -4.2, -4.4, -4.6, -4.8, -5.]);
    expect(a.shape).toEqual([6]);

    const b = dl.linspace(-9, -4, 5);
    expectArraysClose(b, [-9., -7.75, -6.5, -5.25, -4.]);
    expect(b.shape).toEqual([5]);
  });

  it('should throw with no samples', () => {
    expect(() => dl.linspace(2, 10, 0)).toThrow();
  });
});

describeWithFlags('range', ALL_ENVS, () => {
  it('start stop', () => {
    const a = dl.range(0, 3);
    expectArraysEqual(a, [0, 1, 2]);
    expect(a.shape).toEqual([3]);

    const b = dl.range(3, 8);
    expectArraysEqual(b, [3, 4, 5, 6, 7]);
    expect(b.shape).toEqual([5]);
  });

  it('start stop negative', () => {
    const a = dl.range(-2, 3);
    expectArraysEqual(a, [-2, -1, 0, 1, 2]);
    expect(a.shape).toEqual([5]);

    const b = dl.range(4, -2);
    expectArraysEqual(b, [4, 3, 2, 1, 0, -1]);
    expect(b.shape).toEqual([6]);
  });

  it('start stop step', () => {
    const a = dl.range(4, 15, 4);
    expectArraysEqual(a, [4, 8, 12]);
    expect(a.shape).toEqual([3]);

    const b = dl.range(4, 11, 4);
    expectArraysEqual(b, [4, 8]);
    expect(b.shape).toEqual([2]);

    const c = dl.range(4, 17, 4);
    expectArraysEqual(c, [4, 8, 12, 16]);
    expect(c.shape).toEqual([4]);

    const d = dl.range(0, 30, 5);
    expectArraysEqual(d, [0, 5, 10, 15, 20, 25]);
    expect(d.shape).toEqual([6]);

    const e = dl.range(-3, 9, 2);
    expectArraysEqual(e, [-3, -1, 1, 3, 5, 7]);
    expect(e.shape).toEqual([6]);

    const f = dl.range(3, 3);
    expectArraysEqual(f, new Float32Array(0));
    expect(f.shape).toEqual([0]);

    const g = dl.range(3, 3, 1);
    expectArraysEqual(g, new Float32Array(0));
    expect(g.shape).toEqual([0]);

    const h = dl.range(3, 3, 4);
    expectArraysEqual(h, new Float32Array(0));
    expect(h.shape).toEqual([0]);

    const i = dl.range(-18, -2, 5);
    expectArraysEqual(i, [-18, -13, -8, -3]);
    expect(i.shape).toEqual([4]);
  });

  it('start stop large step', () => {
    const a = dl.range(3, 10, 150);
    expectArraysEqual(a, [3]);
    expect(a.shape).toEqual([1]);

    const b = dl.range(10, 500, 205);
    expectArraysEqual(b, [10, 215, 420]);
    expect(b.shape).toEqual([3]);

    const c = dl.range(3, -10, -150);
    expectArraysEqual(c, [3]);
    expect(c.shape).toEqual([1]);

    const d = dl.range(-10, -500, -205);
    expectArraysEqual(d, [-10, -215, -420]);
    expect(d.shape).toEqual([3]);
  });

  it('start stop negative step', () => {
    const a = dl.range(0, -10, -1);
    expectArraysEqual(a, [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]);
    expect(a.shape).toEqual([10]);

    const b = dl.range(0, -10);
    expectArraysEqual(b, [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]);
    expect(b.shape).toEqual([10]);

    const c = dl.range(3, -4, -2);
    expectArraysEqual(c, [3, 1, -1, -3]);
    expect(c.shape).toEqual([4]);

    const d = dl.range(-3, -18, -5);
    expectArraysEqual(d, [-3, -8, -13]);
    expect(d.shape).toEqual([3]);
  });

  it('start stop incompatible step', () => {
    const a = dl.range(3, 10, -2);
    expectArraysEqual(a, new Float32Array(0));
    expect(a.shape).toEqual([0]);

    const b = dl.range(40, 3, 2);
    expectArraysEqual(b, new Float32Array(0));
    expect(b.shape).toEqual([0]);
  });

  it('zero step', () => {
    expect(() => dl.range(2, 10, 0)).toThrow();
  });

  it('should have default dtype', () => {
    const a = dl.range(1, 4);
    expectArraysEqual(a, [1, 2, 3]);
    expect(a.dtype).toEqual('float32');
    expect(a.shape).toEqual([3]);
  });

  it('should have float32 dtype', () => {
    const a = dl.range(1, 4, undefined, 'float32');
    expectArraysEqual(a, [1, 2, 3]);
    expect(a.dtype).toEqual('float32');
    expect(a.shape).toEqual([3]);
  });

  it('should have int32 dtype', () => {
    const a = dl.range(1, 4, undefined, 'int32');
    expectArraysEqual(a, [1, 2, 3]);
    expect(a.dtype).toEqual('int32');
    expect(a.shape).toEqual([3]);
  });
});

describeWithFlags('fill', ALL_ENVS, () => {
  it('1D fill', () => {
    const a = dl.fill([3], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(a, [2, 2, 2]);
  });

  it('2D fill', () => {
    const a = dl.fill([3, 2], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(a, [2, 2, 2, 2, 2, 2]);
  });

  it('3D fill', () => {
    const a = dl.fill([3, 2, 1], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1]);
    expectArraysClose(a, [2, 2, 2, 2, 2, 2]);
  });

  it('4D fill', () => {
    const a = dl.fill([3, 2, 1, 2], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 2]);
    expectArraysClose(a, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
  });
});

describeWithFlags('stack', ALL_ENVS, () => {
  it('scalars 3, 5 and 7', () => {
    const a = dl.scalar(3);
    const b = dl.scalar(5);
    const c = dl.scalar(7);
    const res = dl.stack([a, b, c]);
    expect(res.shape).toEqual([3]);
    expectArraysClose(res, [3, 5, 7]);
  });

  it('scalars 3, 5 and 7 along axis=1 throws error', () => {
    const a = dl.scalar(3);
    const b = dl.scalar(5);
    const c = dl.scalar(7);
    const f = () => dl.stack([a, b, c], 1);
    expect(f).toThrowError();
  });

  it('non matching shapes throws error', () => {
    const a = dl.scalar(3);
    const b = dl.tensor1d([5]);
    const f = () => dl.stack([a, b]);
    expect(f).toThrowError();
  });

  it('non matching dtypes throws error', () => {
    const a = dl.scalar(3);
    const b = dl.scalar(5, 'bool');
    const f = () => dl.stack([a, b]);
    expect(f).toThrowError();
  });

  it('2d but axis=3 throws error', () => {
    const a = dl.zeros([2, 2]);
    const b = dl.zeros([2, 2]);
    const f = () => dl.stack([a, b], 3 /* axis */);
    expect(f).toThrowError();
  });

  it('[1,2], [3,4] and [5,6], axis=0', () => {
    const a = dl.tensor1d([1, 2]);
    const b = dl.tensor1d([3, 4]);
    const c = dl.tensor1d([5, 6]);
    const res = dl.stack([a, b, c], 0 /* axis */);
    expect(res.shape).toEqual([3, 2]);
    expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
  });

  it('[1,2], [3,4] and [5,6], axis=1', () => {
    const a = dl.tensor1d([1, 2]);
    const b = dl.tensor1d([3, 4]);
    const c = dl.tensor1d([5, 6]);
    const res = dl.stack([a, b, c], 1 /* axis */);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(res, [1, 3, 5, 2, 4, 6]);
  });

  it('[[1,2],[3,4]] and [[5, 6], [7, 8]], axis=0', () => {
    const a = dl.tensor2d([[1, 2], [3, 4]]);
    const b = dl.tensor2d([[5, 6], [7, 8]]);
    const res = dl.stack([a, b], 0 /* axis */);
    expect(res.shape).toEqual([2, 2, 2]);
    expectArraysClose(res, [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('[[1,2],[3,4]] and [[5, 6], [7, 8]], axis=2', () => {
    const a = dl.tensor2d([[1, 2], [3, 4]]);
    const b = dl.tensor2d([[5, 6], [7, 8]]);
    const c = dl.tensor2d([[9, 10], [11, 12]]);
    const res = dl.stack([a, b, c], 2 /* axis */);
    expect(res.shape).toEqual([2, 2, 3]);
    expectArraysClose(res, [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]);
  });
});

describeWithFlags('expandDims', ALL_ENVS, () => {
  it('scalar, default axis is 0', () => {
    const res = dl.scalar(1).expandDims();
    expect(res.shape).toEqual([1]);
    expectArraysClose(res, [1]);
  });

  it('scalar, axis is out of bounds throws error', () => {
    const f = () => dl.scalar(1).expandDims(1);
    expect(f).toThrowError();
  });

  it('1d, axis=0', () => {
    const res = dl.tensor1d([1, 2, 3]).expandDims(0 /* axis */);
    expect(res.shape).toEqual([1, 3]);
    expectArraysClose(res, [1, 2, 3]);
  });

  it('1d, axis=1', () => {
    const res = dl.tensor1d([1, 2, 3]).expandDims(1 /* axis */);
    expect(res.shape).toEqual([3, 1]);
    expectArraysClose(res, [1, 2, 3]);
  });

  it('2d, axis=0', () => {
    const res = dl.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(0 /* axis */);
    expect(res.shape).toEqual([1, 3, 2]);
    expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
  });

  it('2d, axis=1', () => {
    const res = dl.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(1 /* axis */);
    expect(res.shape).toEqual([3, 1, 2]);
    expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
  });

  it('2d, axis=2', () => {
    const res = dl.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(2 /* axis */);
    expect(res.shape).toEqual([3, 2, 1]);
    expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
  });
});
