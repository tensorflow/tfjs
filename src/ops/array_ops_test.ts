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

import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, expectArraysClose, expectArraysEqual, expectPromiseToFail, expectValuesInRange, WEBGL_ENVS} from '../test_util';
import * as util from '../util';

import {expectArrayInMeanStdRange, jarqueBeraNormalityTest} from './rand_util';

describeWithFlags('zeros', ALL_ENVS, () => {
  it('1D default dtype', () => {
    const a: tf.Tensor1D = tf.zeros([3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(a, [0, 0, 0]);
  });

  it('1D float32 dtype', () => {
    const a = tf.zeros([3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(a, [0, 0, 0]);
  });

  it('1D int32 dtype', () => {
    const a = tf.zeros([3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(a, [0, 0, 0]);
  });

  it('1D bool dtype', () => {
    const a = tf.zeros([3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(a, [0, 0, 0]);
  });

  it('2D default dtype', () => {
    const a = tf.zeros([3, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('2D float32 dtype', () => {
    const a = tf.zeros([3, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('2D int32 dtype', () => {
    const a = tf.zeros([3, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });

  it('2D bool dtype', () => {
    const a = tf.zeros([3, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2]);
    expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });

  it('3D default dtype', () => {
    const a = tf.zeros([2, 2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysClose(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D float32 dtype', () => {
    const a = tf.zeros([2, 2, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysClose(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D int32 dtype', () => {
    const a = tf.zeros([2, 2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysEqual(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D bool dtype', () => {
    const a = tf.zeros([2, 2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysEqual(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('4D default dtype', () => {
    const a = tf.zeros([3, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('4D float32 dtype', () => {
    const a = tf.zeros([3, 2, 1, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('4D int32 dtype', () => {
    const a = tf.zeros([3, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });

  it('4D bool dtype', () => {
    const a = tf.zeros([3, 2, 1, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });
});

describeWithFlags('ones', ALL_ENVS, () => {
  it('1D default dtype', () => {
    const a = tf.ones([3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(a, [1, 1, 1]);
  });

  it('1D float32 dtype', () => {
    const a = tf.ones([3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(a, [1, 1, 1]);
  });

  it('1D int32 dtype', () => {
    const a = tf.ones([3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(a, [1, 1, 1]);
  });

  it('1D bool dtype', () => {
    const a = tf.ones([3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(a, [1, 1, 1]);
  });

  it('2D default dtype', () => {
    const a = tf.ones([3, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('2D float32 dtype', () => {
    const a = tf.ones([3, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('2D int32 dtype', () => {
    const a = tf.ones([3, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });

  it('2D bool dtype', () => {
    const a = tf.ones([3, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2]);
    expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });

  it('3D default dtype', () => {
    const a = tf.ones([2, 2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysClose(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('3D float32 dtype', () => {
    const a = tf.ones([2, 2, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysClose(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('3D int32 dtype', () => {
    const a = tf.ones([2, 2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysEqual(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('3D bool dtype', () => {
    const a = tf.ones([2, 2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysEqual(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('4D default dtype', () => {
    const a = tf.ones([3, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('4D float32 dtype', () => {
    const a = tf.ones([3, 2, 1, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('4D int32 dtype', () => {
    const a = tf.ones([3, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });

  it('4D bool dtype', () => {
    const a = tf.ones([3, 2, 1, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });
});

describeWithFlags('zerosLike', ALL_ENVS, () => {
  it('1D default dtype', () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(b, [0, 0, 0]);
  });

  it('1D float32 dtype', () => {
    const a = tf.tensor1d([1, 2, 3], 'float32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(b, [0, 0, 0]);
  });

  it('1D int32 dtype', () => {
    const a = tf.tensor1d([1, 2, 3], 'int32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(b, [0, 0, 0]);
  });

  it('1D bool dtype', () => {
    const a = tf.tensor1d([1, 2, 3], 'bool');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(b, [0, 0, 0]);
  });

  it('2D default dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('2D float32 dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('2D int32 dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('2D bool dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'bool');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('3D default dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('3D float32 dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('3D int32 dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'int32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('3D bool dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'bool');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('4D default dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('4D float32 dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('4D int32 dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'int32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('4D bool dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'bool');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('4D default dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('5D float32 dtype', () => {
    const a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'float32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 2, 2, 1, 1]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('5D int32 dtype', () => {
    const a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'int32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([1, 2, 2, 1, 1]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('5D bool dtype', () => {
    const a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'bool');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([1, 2, 2, 1, 1]);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('5D default dtype', () => {
    const a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 2, 2, 1, 1]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('6D float32 dtype', () => {
    const a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'float32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 2, 2, 1, 1, 1]);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('6D int32 dtype', () => {
    const a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'int32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual(a.shape);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('6D bool dtype', () => {
    const a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'bool');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual(a.shape);
    expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('6D default dtype', () => {
    const a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual(a.shape);
    expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.zerosLike({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'zerosLike' must be a Tensor/);
  });
});

describeWithFlags('onesLike', ALL_ENVS, () => {
  it('1D default dtype', () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(b, [1, 1, 1]);
  });

  it('1D float32 dtype', () => {
    const a = tf.tensor1d([1, 2, 3], 'float32');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(b, [1, 1, 1]);
  });

  it('1D int32 dtype', () => {
    const a = tf.tensor1d([1, 2, 3], 'int32');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(b, [1, 1, 1]);
  });

  it('1D bool dtype', () => {
    const a = tf.tensor1d([1, 2, 3], 'bool');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(b, [1, 1, 1]);
  });

  it('2D default dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('2D float32 dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('2D int32 dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('2D bool dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'bool');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('3D default dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('3D float32 dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('3D int32 dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'int32');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('3D bool dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'bool');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('4D default dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('4D float32 dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('4D int32 dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'int32');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('4D bool dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'bool');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('4D default dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('5D float32 dtype', () => {
    const a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'float32');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 2, 2, 1, 1]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('5D int32 dtype', () => {
    const a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'int32');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([1, 2, 2, 1, 1]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('5D bool dtype', () => {
    const a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'bool');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([1, 2, 2, 1, 1]);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('5D default dtype', () => {
    const a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1]);
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 2, 2, 1, 1]);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('6D int32 dtype', () => {
    const a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'int32');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual(a.shape);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('6D bool dtype', () => {
    const a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'bool');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual(a.shape);
    expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('6D default dtype', () => {
    const a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1]);
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual(a.shape);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('6D float32 dtype', () => {
    const a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'float32');
    const b = tf.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual(a.shape);
    expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.onesLike({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'onesLike' must be a Tensor/);
  });
});

describeWithFlags('rand', ALL_ENVS, () => {
  it('should return a random 1D float32 array', () => {
    const shape: [number] = [10];

    // Enusre defaults to float32 w/o type:
    let result = tf.rand(shape, () => util.randUniform(0, 2));
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2);

    result = tf.rand(shape, () => util.randUniform(0, 1.5));
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 1D int32 array', () => {
    const shape: [number] = [10];
    const result = tf.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 1D bool array', () => {
    const shape: [number] = [10];
    const result = tf.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 2D float32 array', () => {
    const shape = [3, 4];

    // Enusre defaults to float32 w/o type:
    let result = tf.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = tf.rand(shape, () => util.randUniform(0, 1.5), 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 2D int32 array', () => {
    const shape = [3, 4];
    const result = tf.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 2D bool array', () => {
    const shape = [3, 4];
    const result = tf.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 3D float32 array', () => {
    const shape = [3, 4, 5];

    // Enusre defaults to float32 w/o type:
    let result = tf.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = tf.rand(shape, () => util.randUniform(0, 1.5), 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 3D int32 array', () => {
    const shape = [3, 4, 5];
    const result = tf.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 3D bool array', () => {
    const shape = [3, 4, 5];
    const result = tf.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 4D float32 array', () => {
    const shape = [3, 4, 5, 6];

    // Enusre defaults to float32 w/o type:
    let result = tf.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = tf.rand(shape, () => util.randUniform(0, 1.5));
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 4D int32 array', () => {
    const shape = [3, 4, 5, 6];
    const result = tf.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 4D bool array', () => {
    const shape = [3, 4, 5, 6];
    const result = tf.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });
});

describeWithFlags('eye', ALL_ENVS, () => {
  it('1x1', () => {
    expectArraysClose(tf.eye(1), tf.tensor2d([[1]]));
  });

  it('2x2', () => {
    expectArraysClose(tf.eye(2), tf.tensor2d([[1, 0], [0, 1]]));
  });

  it('3x3', () => {
    expectArraysClose(
        tf.eye(3), tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
  });

  it('3x4', () => {
    expectArraysClose(
        tf.eye(3, 4), tf.tensor2d([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]));
  });

  it('4x3', () => {
    expectArraysClose(
        tf.eye(4, 3),
        tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]));
  });

  it('with 1D batchShape', () => {
    expectArraysClose(
        tf.eye(2, 2, [3]),
        tf.tensor3d([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]]));
  });

  it('with 2D batchShape', () => {
    expectArraysClose(tf.eye(2, 2, [2, 3]), tf.tensor4d([
      [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]],
      [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]]
    ]));
  });

  it('3x3, int32', () => {
    expectArraysClose(
        tf.eye(3, 3, null, 'int32'),
        tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [3, 3], 'int32'));
  });

  it('3x3, bool', () => {
    expectArraysClose(
        tf.eye(3, 3, null, 'bool'),
        tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [3, 3], 'bool'));
  });
});

describeWithFlags('randomNormal', ALL_ENVS, () => {
  const SEED = 2002;
  const EPSILON = 0.05;

  it('should return a float32 1D of random normal values', () => {
    const SAMPLES = 10000;

    // Ensure defaults to float32.
    let result = tf.randomNormal([SAMPLES], 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);

    result = tf.randomNormal([SAMPLES], 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
  });

  it('should return a int32 1D of random normal values', () => {
    const SAMPLES = 10000;
    const result = tf.randomNormal([SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a float32 2D of random normal values', () => {
    const SAMPLES = 100;

    // Ensure defaults to float32.
    let result = tf.randomNormal([SAMPLES, SAMPLES], 0, 2.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 2.5, EPSILON);

    result = tf.randomNormal([SAMPLES, SAMPLES], 0, 3.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
  });

  it('should return a int32 2D of random normal values', () => {
    const SAMPLES = 100;
    const result = tf.randomNormal([SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a float32 3D of random normal values', () => {
    const SAMPLES_SHAPE = [20, 20, 20];

    // Ensure defaults to float32.
    let result = tf.randomNormal(SAMPLES_SHAPE, 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);

    result = tf.randomNormal(SAMPLES_SHAPE, 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
  });

  it('should return a int32 3D of random normal values', () => {
    const SAMPLES_SHAPE = [20, 20, 20];
    const result = tf.randomNormal(SAMPLES_SHAPE, 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a float32 4D of random normal values', () => {
    const SAMPLES_SHAPE = [10, 10, 10, 10];

    // Ensure defaults to float32.
    let result = tf.randomNormal(SAMPLES_SHAPE, 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);

    result = tf.randomNormal(SAMPLES_SHAPE, 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
  });

  it('should return a int32 4D of random normal values', () => {
    const SAMPLES_SHAPE = [10, 10, 10, 10];

    const result = tf.randomNormal(SAMPLES_SHAPE, 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a int32 5D of random normal values', () => {
    const SAMPLES_SHAPE = [10, 10, 10, 10, 10];

    const result = tf.randomNormal(SAMPLES_SHAPE, 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(result);
    expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });
});

describeWithFlags('truncatedNormal', ALL_ENVS, () => {
  // Expect slightly higher variances for truncated values.
  const EPSILON = 0.60;
  const SEED = 2002;

  function assertTruncatedValues(array: tf.Tensor, mean: number, stdv: number) {
    const bounds = mean + stdv * 2;
    const values = array.dataSync();
    for (let i = 0; i < values.length; i++) {
      expect(Math.abs(values[i])).toBeLessThanOrEqual(bounds);
    }
  }

  it('should return a random 1D float32 array', () => {
    const shape: [number] = [1000];

    // Ensure defaults to float32 w/o type:
    let result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a randon 1D int32 array', () => {
    const shape: [number] = [1000];
    const result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });

  it('should return a 2D float32 array', () => {
    const shape: [number, number] = [50, 50];

    // Ensure defaults to float32 w/o type:
    let result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a 2D int32 array', () => {
    const shape: [number, number] = [50, 50];
    const result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });

  it('should return a 3D float32 array', () => {
    const shape: [number, number, number] = [10, 10, 10];

    // Ensure defaults to float32 w/o type:
    let result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a 3D int32 array', () => {
    const shape: [number, number, number] = [10, 10, 10];
    const result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });

  it('should return a 4D float32 array', () => {
    const shape: [number, number, number, number] = [5, 5, 5, 5];

    // Ensure defaults to float32 w/o type:
    let result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a 4D int32 array', () => {
    const shape: [number, number, number, number] = [5, 5, 5, 5];
    const result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });
});

describeWithFlags('randomUniform', ALL_ENVS, () => {
  it('should return a random 1D float32 array', () => {
    const shape: [number] = [10];

    // Enusre defaults to float32 w/o type:
    let result = tf.randomUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = tf.randomUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 1D int32 array', () => {
    const shape: [number] = [10];
    const result = tf.randomUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 1D bool array', () => {
    const shape: [number] = [10];
    const result = tf.randomUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 2D float32 array', () => {
    const shape: [number, number] = [3, 4];

    // Enusre defaults to float32 w/o type:
    let result = tf.randomUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = tf.randomUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 2D int32 array', () => {
    const shape: [number, number] = [3, 4];
    const result = tf.randomUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 2D bool array', () => {
    const shape: [number, number] = [3, 4];
    const result = tf.randomUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 3D float32 array', () => {
    const shape: [number, number, number] = [3, 4, 5];

    // Enusre defaults to float32 w/o type:
    let result = tf.randomUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = tf.randomUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 3D int32 array', () => {
    const shape: [number, number, number] = [3, 4, 5];
    const result = tf.randomUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 3D bool array', () => {
    const shape: [number, number, number] = [3, 4, 5];
    const result = tf.randomUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 4D float32 array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];

    // Enusre defaults to float32 w/o type:
    let result = tf.randomUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = tf.randomUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 4D int32 array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];
    const result = tf.randomUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 4D bool array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];
    const result = tf.randomUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });

  it('should return a random 5D float32 array', () => {
    const shape: [number, number, number, number, number] = [2, 3, 4, 5, 6];

    // Enusre defaults to float32 w/o type:
    let result = tf.randomUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 2.5);

    result = tf.randomUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 5D int32 array', () => {
    const shape: [number, number, number, number, number] = [2, 3, 4, 5, 6];
    const result = tf.randomUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(result, 0, 2);
  });

  it('should return a random 5D bool array', () => {
    const shape: [number, number, number, number, number] = [2, 3, 4, 5, 6];
    const result = tf.randomUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    expectValuesInRange(result, 0, 1);
  });
});

describeWithFlags('fromPixels', WEBGL_ENVS, () => {
  it('ImageData 1x1x3', () => {
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 0;
    pixels.data[1] = 80;
    pixels.data[2] = 160;
    pixels.data[3] = 240;

    const array = tf.fromPixels(pixels, 3);

    expectArraysEqual(array, [0, 80, 160]);
  });

  it('ImageData 1x1x4', () => {
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 0;
    pixels.data[1] = 80;
    pixels.data[2] = 160;
    pixels.data[3] = 240;

    const array = tf.fromPixels(pixels, 4);

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

    const array = tf.fromPixels(pixels, 3);

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

    const array = tf.fromPixels(pixels, 4);

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
    const res = tf.fromPixels(pixels, 3);
    expect(res.shape).toEqual([2, 1, 3]);
    expect(res.dtype).toBe('int32');
    expectArraysClose(res, [2, 3, 4, 5, 6, 7]);
  });

  it('fromPixels, reshape, then do tf.add()', () => {
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 2;
    pixels.data[1] = 3;
    pixels.data[2] = 4;
    pixels.data[3] = 255;  // Not used.
    const a = tf.fromPixels(pixels, 3).reshape([1, 1, 1, 3]);
    const res = a.add(tf.scalar(2, 'int32'));
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
    const a = tf.fromPixels(pixelsA, 3).toFloat();
    const b = tf.fromPixels(pixelsB, 3).toFloat();
    const res = a.add(b);
    expect(res.shape).toEqual([1, 1, 3]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(res, [260, 9, 11]);
  });
});

describeWithFlags('toPixels no canvas', ALL_ENVS, () => {
  it('draws a rank-2 float32 tensor', done => {
    const x = tf.tensor2d([.1, .2], [2, 1], 'float32');

    tf.toPixels(x).then(data => {
      const expected = new Uint8ClampedArray([
        Math.round(.1 * 255), Math.round(.1 * 255), Math.round(.1 * 255), 255,
        Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255), 255
      ]);
      expect(data).toEqual(expected);

      done();
    });
  });

  it('draws a rank-2 int32 tensor', done => {
    const x = tf.tensor2d([10, 20], [2, 1], 'int32');

    tf.toPixels(x).then(data => {
      const expected =
          new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
      expect(data).toEqual(expected);

      done();
    });
  });

  it('draws a rank-3 float32 tensor, 1 channel', done => {
    const x = tf.tensor3d([.1, .2], [2, 1, 1], 'float32');

    tf.toPixels(x).then(data => {
      const expected = new Uint8ClampedArray([
        Math.round(.1 * 255), Math.round(.1 * 255), Math.round(.1 * 255), 255,
        Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255), 255
      ]);
      expect(data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 int32 tensor, 1 channel', done => {
    const x = tf.tensor3d([10, 20], [2, 1, 1], 'int32');

    tf.toPixels(x).then(data => {
      const expected =
          new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
      expect(data).toEqual(expected);

      done();
    });
  });

  it('draws a rank-3 float32 tensor, 3 channel', done => {
    const x = tf.tensor3d([.05, .1, .15, .2, .25, .3], [2, 1, 3], 'float32');

    tf.toPixels(x).then(data => {
      const expected = new Uint8ClampedArray([
        Math.round(.05 * 255), Math.round(.1 * 255), Math.round(.15 * 255), 255,
        Math.round(.2 * 255), Math.round(.25 * 255), Math.round(.30 * 255), 255
      ]);
      expect(data).toEqual(expected);

      done();
    });
  });

  it('draws a rank-3 int32 tensor, 3 channel', done => {
    const x = tf.tensor3d([10, 20, 30, 40, 50, 60], [2, 1, 3], 'int32');

    tf.toPixels(x).then(data => {
      const expected =
          new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]);
      expect(data).toEqual(expected);

      done();
    });
  });

  it('draws a rank-3 float32 tensor, 4 channel', done => {
    const x =
        tf.tensor3d([.05, .1, .15, .2, .25, .3, .35, .4], [2, 1, 4], 'float32');

    tf.toPixels(x).then(data => {
      const expected = new Uint8ClampedArray([
        Math.round(.05 * 255), Math.round(.1 * 255), Math.round(.15 * 255),
        Math.round(.20 * 255), Math.round(.25 * 255), Math.round(.30 * 255),
        Math.round(.35 * 255), Math.round(.4 * 255)
      ]);
      expect(data).toEqual(expected);

      done();
    });
  });

  it('draws a rank-3 int32 tensor, 4 channel', done => {
    const x = tf.tensor3d([10, 20, 30, 40, 50, 60, 70, 80], [2, 1, 4], 'int32');

    tf.toPixels(x).then(data => {
      const expected = new Uint8ClampedArray([10, 20, 30, 40, 50, 60, 70, 80]);
      expect(data).toEqual(expected);

      done();
    });
  });

  it('throws for scalars', done => {
    // tslint:disable-next-line:no-any
    expectPromiseToFail(() => tf.toPixels(tf.scalar(1) as any), done);
  });

  it('throws for rank-1 tensors', done => {
    // tslint:disable-next-line:no-any
    expectPromiseToFail(() => tf.toPixels(tf.tensor1d([1]) as any), done);
  });
  it('throws for rank-4 tensors', done => {
    expectPromiseToFail(
        // tslint:disable-next-line:no-any
        () => tf.toPixels(tf.tensor4d([1], [1, 1, 1, 1]) as any), done);
  });
  it('throws for bool dtype', done => {
    expectPromiseToFail(
        () => tf.toPixels(tf.tensor2d([1], [1, 1], 'bool')), done);
  });
  it('throws for rank-3 depth = 2', done => {
    expectPromiseToFail(
        () => tf.toPixels(tf.tensor3d([1, 2], [1, 1, 2])), done);
  });
  it('throws for rank-3 depth = 5', done => {
    expectPromiseToFail(
        () => tf.toPixels(tf.tensor3d([1, 2, 3, 4, 5], [1, 1, 5])), done);
  });
  it('throws for float32 tensor with values not in [0 - 1]', done => {
    expectPromiseToFail(() => tf.toPixels(tf.tensor2d([-1, .5], [1, 2])), done);
  });
  it('throws for int32 tensor with values not in [0 - 255]', done => {
    expectPromiseToFail(
        () => tf.toPixels(tf.tensor2d([-1, 100], [1, 2], 'int32')), done);
  });
  it('throws when passed a non-tensor', done => {
    // tslint:disable-next-line:no-any
    expectPromiseToFail(() => tf.toPixels({} as any), done);
  });
});

describeWithFlags('toPixels', WEBGL_ENVS, () => {
  it('draws a rank-2 float32 tensor, canvas', done => {
    const x = tf.tensor2d([.1, .2], [2, 1], 'float32');
    const canvas = document.createElement('canvas');

    tf.toPixels(x, canvas).then(data => {
      const expected = new Uint8ClampedArray([
        Math.round(.1 * 255), Math.round(.1 * 255), Math.round(.1 * 255), 255,
        Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255), 255
      ]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-2 int32 tensor, canvas', done => {
    const x = tf.tensor2d([10, 20], [2, 1], 'int32');
    const canvas = document.createElement('canvas');

    tf.toPixels(x, canvas).then(data => {
      const expected =
          new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 float32 tensor, 1 channel, canvas', done => {
    const x = tf.tensor3d([.1, .2], [2, 1, 1], 'float32');
    const canvas = document.createElement('canvas');

    tf.toPixels(x, canvas).then(data => {
      const expected = new Uint8ClampedArray([
        Math.round(.1 * 255), Math.round(.1 * 255), Math.round(.1 * 255), 255,
        Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255), 255
      ]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 int32 tensor, 1 channel, canvas', done => {
    const x = tf.tensor3d([10, 20], [2, 1, 1], 'int32');
    const canvas = document.createElement('canvas');

    tf.toPixels(x, canvas).then(data => {
      const expected =
          new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 float32 tensor, 3 channel, canvas', done => {
    const x = tf.tensor3d([.05, .1, .15, .20, .25, .30], [2, 1, 3], 'float32');
    const canvas = document.createElement('canvas');

    tf.toPixels(x, canvas).then(data => {
      const expected = new Uint8ClampedArray([
        Math.round(.05 * 255), Math.round(.1 * 255), Math.round(.15 * 255), 255,
        Math.round(.2 * 255), Math.round(.25 * 255), Math.round(.3 * 255), 255
      ]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 int32 tensor, 3 channel, canvas', done => {
    const x = tf.tensor3d([10, 20, 30, 40, 50, 60], [2, 1, 3], 'int32');
    const canvas = document.createElement('canvas');

    tf.toPixels(x, canvas).then(data => {
      const expected =
          new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);
      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 float32 tensor, 4 channel, canvas', done => {
    // ImageData roundtrips are lossy because of pre-multiplied alphas, so we
    // use an alpha = 1 to avoid losing precision on r, g, b channels in these
    // tests https://www.w3.org/TR/2dcontext/
    const x =
        tf.tensor3d([.05, .1, .15, 1, .20, .25, .30, 1], [2, 1, 4], 'float32');
    const canvas = document.createElement('canvas');

    tf.toPixels(x, canvas).then(data => {
      const expected = new Uint8ClampedArray([
        Math.round(.05 * 255), Math.round(.1 * 255), Math.round(.15 * 255), 255,
        Math.round(.20 * 255), Math.round(.25 * 255), Math.round(.30 * 255), 255
      ]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 int32 tensor, 4 channel, canvas', done => {
    // ImageData roundtrips are lossy because of pre-multiplied alphas, so we
    // use an alpha = 1 to avoid losing precision on r, g, b channels in these
    // tests https://www.w3.org/TR/2dcontext/
    const x =
        tf.tensor3d([10, 20, 30, 255, 50, 60, 70, 255], [2, 1, 4], 'int32');
    const canvas = document.createElement('canvas');

    tf.toPixels(x, canvas).then(data => {
      const expected =
          new Uint8ClampedArray([10, 20, 30, 255, 50, 60, 70, 255]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });
});

describeWithFlags('clone', ALL_ENVS, () => {
  it('1D default dtype', () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(b, [1, 2, 3]);
  });

  it('1D float32 dtype', () => {
    const a = tf.tensor1d([1, 2, 3], 'float32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(b, [1, 2, 3]);
  });

  it('1D int32 dtype', () => {
    const a = tf.tensor1d([1, 2, 3], 'int32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(b, [1, 2, 3]);
  });

  it('1D bool dtype', () => {
    const a = tf.tensor1d([1, 1, 0], 'bool');
    const b = tf.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(b, [1, 1, 0]);
  });

  it('2D default dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('2D float32 dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('2D int32 dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(b, [1, 2, 3, 4]);
  });

  it('2D bool dtype', () => {
    const a = tf.tensor2d([1, 1, 1, 0], [2, 2], 'bool');
    const b = tf.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(b, [1, 1, 1, 0]);
  });

  it('3D default dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('3D float32 dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('3D int32 dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'int32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(b, [1, 2, 3, 4]);
  });

  it('3D bool dtype', () => {
    const a = tf.tensor3d([1, 1, 1, 0], [2, 2, 1], 'bool');
    const b = tf.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(b, [1, 1, 1, 0]);
  });

  it('4D default dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('4D float32 dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('4D int32 dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'int32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(b, [1, 2, 3, 4]);
  });

  it('4D bool dtype', () => {
    const a = tf.tensor4d([1, 1, 1, 0], [2, 2, 1, 1], 'bool');
    const b = tf.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(b, [1, 1, 1, 0]);
  });

  it('gradient: 1D', () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([4, 5, 6]);
    const da = tf.grad(x => tf.clone(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([3]);
    expectArraysClose(da, [4, 5, 6]);
  });

  it('gradient: 2D int32', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const dy = tf.tensor2d([5, 6, 7, 8], [2, 2], 'float32');
    const da = tf.grad(x => tf.clone(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 2]);
    expectArraysEqual(da, [5, 6, 7, 8]);
  });

  it('gradient: 4D bool', () => {
    const a = tf.tensor4d([1, 1, 1, 0], [2, 2, 1, 1], 'bool');
    const dy = tf.tensor4d([5, 6, 7, 8], [2, 2, 1, 1], 'float32');
    const da = tf.grad(x => tf.clone(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(da, [5, 6, 7, 8]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.clone({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'clone' must be a Tensor/);
  });
});

describeWithFlags('tile', ALL_ENVS, () => {
  it('1D (tile)', () => {
    const t = tf.tensor1d([1, 2, 3]);
    const t2 = tf.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expectArraysClose(t2, [1, 2, 3, 1, 2, 3]);
  });

  it('2D (tile)', () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    let t2 = tf.tile(t, [1, 2]);

    expect(t2.shape).toEqual([2, 4]);
    expectArraysClose(t2, [1, 11, 1, 11, 2, 22, 2, 22]);

    t2 = tf.tile(t, [2, 1]);
    expect(t2.shape).toEqual([4, 2]);
    expectArraysClose(t2, [1, 11, 2, 22, 1, 11, 2, 22]);

    t2 = tf.tile(t, [2, 2]);
    expect(t2.shape).toEqual([4, 4]);
    expectArraysClose(
        t2, [1, 11, 1, 11, 2, 22, 2, 22, 1, 11, 1, 11, 2, 22, 2, 22]);
  });

  it('3D (tile)', () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const t2 = tf.tile(t, [1, 2, 1]);

    expect(t2.shape).toEqual([2, 4, 2]);
    expectArraysClose(t2, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
  });

  it('4D (tile)', () => {
    const t = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2]);
    const t2 = tf.tile(t, [1, 2, 1, 1]);

    expect(t2.shape).toEqual([1, 4, 2, 2]);
    expectArraysClose(t2, [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('5D (tile)', () => {
    const t = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 2, 2]);
    const t2 = tf.tile(t, [1, 2, 1, 1, 1]);

    expect(t2.shape).toEqual([1, 2, 2, 2, 2]);
    expectArraysClose(t2, [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('propagates NaNs', () => {
    const t = tf.tensor1d([1, 2, NaN]);

    const t2 = tf.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expectArraysClose(t2, [1, 2, NaN, 1, 2, NaN]);
  });

  it('1D bool (tile)', () => {
    const t = tf.tensor1d([true, false, true], 'bool');
    const t2 = tf.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(t2, [1, 0, 1, 1, 0, 1]);
  });

  it('2D bool (tile)', () => {
    const t = tf.tensor2d([true, false, true, true], [2, 2], 'bool');
    let t2 = tf.tile(t, [1, 2]);

    expect(t2.shape).toEqual([2, 4]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(t2, [1, 0, 1, 0, 1, 1, 1, 1]);

    t2 = tf.tile(t, [2, 1]);
    expect(t2.shape).toEqual([4, 2]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(t2, [1, 0, 1, 1, 1, 0, 1, 1]);

    t2 = tf.tile(t, [2, 2]);
    expect(t2.shape).toEqual([4, 4]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(t2, [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]);
  });

  it('3D bool (tile)', () => {
    const t = tf.tensor3d(
        [true, false, true, false, true, false, true, false], [2, 2, 2],
        'bool');
    const t2 = tf.tile(t, [1, 2, 1]);

    expect(t2.shape).toEqual([2, 4, 2]);
    expect(t2.dtype).toBe('bool');
    expectArraysEqual(t2, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
  });

  it('1D int32 (tile)', () => {
    const t = tf.tensor1d([1, 2, 5], 'int32');
    const t2 = tf.tile(t, [2]);

    expect(t2.shape).toEqual([6]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(t2, [1, 2, 5, 1, 2, 5]);
  });

  it('2D int32 (tile)', () => {
    const t = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    let t2 = tf.tile(t, [1, 2]);

    expect(t2.shape).toEqual([2, 4]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(t2, [1, 2, 1, 2, 3, 4, 3, 4]);

    t2 = tf.tile(t, [2, 1]);
    expect(t2.shape).toEqual([4, 2]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(t2, [1, 2, 3, 4, 1, 2, 3, 4]);

    t2 = tf.tile(t, [2, 2]);
    expect(t2.shape).toEqual([4, 4]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(t2, [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4]);
  });

  it('3D int32 (tile)', () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2], 'int32');
    const t2 = tf.tile(t, [1, 2, 1]);

    expect(t2.shape).toEqual([2, 4, 2]);
    expect(t2.dtype).toBe('int32');
    expectArraysEqual(t2, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
  });

  it('1D (tile) gradient', () => {
    const x = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([0.1, 0.2, 0.3, 1, 2, 3, 10, 20, 30]);
    const gradients = tf.grad(x => tf.tile(x, [3]))(x, dy);
    expectArraysClose(gradients, tf.tensor1d([11.1, 22.2, 33.3]));
  });

  it('2D (tile) gradient', () => {
    const x = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const dy = tf.tensor2d([[1, 2, 10, 20], [3, 4, 30, 40]], [2, 4]);
    const gradients = tf.grad(x => tf.tile(x, [1, 2]))(x, dy);
    expectArraysClose(gradients, tf.tensor2d([[11, 22], [33, 44]], [2, 2]));
  });

  it('3D (tile) gradient', () => {
    const x = tf.tensor3d([[[1], [2]], [[3], [4]]], [2, 2, 1]);
    const dy = tf.tensor3d([[[1, 10], [2, 20]], [[3, 30], [4, 40]]], [2, 2, 2]);
    const gradients = tf.grad(x => tf.tile(x, [1, 1, 2]))(x, dy);
    expectArraysClose(
        gradients, tf.tensor3d([[[11], [22]], [[33], [44]]], [2, 2, 1]));
  });

  it('4D (tile) gradient', () => {
    const x = tf.tensor4d([[[[1]], [[2]]], [[[3]], [[4]]]], [2, 2, 1, 1]);
    const dy = tf.tensor4d(
        [
          [[[.01, .1], [1, 10]], [[.02, .2], [2, 20]]],
          [[[.03, .3], [3, 30]], [[.04, .4], [4, 40]]]
        ],
        [2, 2, 2, 2]);
    const gradients = tf.grad(x => tf.tile(x, [1, 1, 2, 2]))(x, dy);
    expectArraysClose(
        gradients,
        tf.tensor4d(
            [[[[11.11]], [[22.22]]], [[[33.33]], [[44.44]]]], [2, 2, 1, 1]));
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.tile({} as tf.Tensor, [1]))
        .toThrowError(/Argument 'x' passed to 'tile' must be a Tensor/);
  });
});

describeWithFlags('gather', ALL_ENVS, () => {
  it('1D (gather)', () => {
    const t = tf.tensor1d([1, 2, 3]);

    const t2 = tf.gather(t, tf.tensor1d([0, 2, 0, 1], 'int32'), 0);

    expect(t2.shape).toEqual([4]);
    expectArraysClose(t2, [1, 3, 1, 2]);
  });

  it('2D (gather)', () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    let t2 = tf.gather(t, tf.tensor1d([1, 0, 0, 1], 'int32'), 0);
    expect(t2.shape).toEqual([4, 2]);
    expectArraysClose(t2, [2, 22, 1, 11, 1, 11, 2, 22]);

    t2 = tf.gather(t, tf.tensor1d([1, 0, 0, 1], 'int32'), 1);
    expect(t2.shape).toEqual([2, 4]);
    expectArraysClose(t2, [11, 1, 1, 11, 22, 2, 2, 22]);
  });

  it('3D (gather)', () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);

    const t2 = tf.gather(t, tf.tensor1d([1, 0, 0, 1], 'int32'), 2);

    expect(t2.shape).toEqual([2, 2, 4]);
    expectArraysClose(t2, [2, 1, 1, 2, 4, 3, 3, 4, 6, 5, 5, 6, 8, 7, 7, 8]);
  });

  it('bool (gather)', () => {
    const t = tf.tensor1d([true, false, true], 'bool');

    const t2 = tf.gather(t, tf.tensor1d([0, 2, 0, 1], 'int32'), 0);

    expect(t2.shape).toEqual([4]);
    expect(t2.dtype).toBe('bool');
    expect(t2.dataSync()).toEqual(new Uint8Array([1, 1, 1, 0]));
  });

  it('int32 (gather)', () => {
    const t = tf.tensor1d([1, 2, 5], 'int32');

    const t2 = tf.gather(t, tf.tensor1d([0, 2, 0, 1], 'int32'), 0);

    expect(t2.shape).toEqual([4]);
    expect(t2.dtype).toBe('int32');
    expect(t2.dataSync()).toEqual(new Int32Array([1, 5, 1, 2]));
  });

  it('propagates NaNs', () => {
    const t = tf.tensor1d([1, 2, NaN]);

    const t2 = tf.gather(t, tf.tensor1d([0, 2, 0, 1], 'int32'), 0);

    expect(t2.shape).toEqual([4]);
    expectArraysClose(t2, [1, NaN, 1, 2]);
  });

  it('chaining, axis=1', () => {
    const x = tf.zeros([2, 4, 6]);
    // [0, 2, 4]
    const indices = tf.range(0, 6, 2, 'int32');
    const axis = 2;
    expect(x.gather(indices, axis).shape).toEqual([2, 4, 3]);
  });

  it('indices not int32 throws error', () => {
    const x = tf.zeros([2, 4, 6]);
    // [0, 2, 4]
    const indices = tf.range(0, 6, 2);
    const axis = 2;
    expect(() => x.gather(indices, axis)).toThrowError();
  });

  it('throws when passed x as a non-tensor', () => {
    expect(() => tf.gather({} as tf.Tensor, tf.tensor1d([1])))
        .toThrowError(/Argument 'x' passed to 'gather' must be a Tensor/);
  });

  it('throws when passed indices as a non-tensor', () => {
    // tslint:disable-next-line:no-any
    expect(() => tf.gather(tf.tensor1d([1]), {} as any))
        .toThrowError(/Argument 'indices' passed to 'gather' must be a Tensor/);
  });

  it('gradient 1D (gather)', () => {
    const t = tf.tensor1d([1, 2, 3]);
    const indices = tf.tensor1d([0, 2, 0, 1], 'int32');
    const dy = tf.tensor([3, 4, 5, 6]);

    const gradients = tf.grad(t => tf.gather(t, indices))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(gradients, [8, 6, 4]);
  });

  it('gradient 2D (gather) axis=0 shape=[2, 2]', () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    const indices = tf.tensor1d([1, 0, 0, 1], 'int32');
    const dy = tf.tensor([3, 4, 5, 6, 7, 8, 9, 10], [4, 2]);
    const axis = 0;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(gradients, [12, 14, 12, 14]);
  });

  it('gradient 2D (gather) axis=0 shape=[4, 1]', () => {
    const t = tf.tensor2d([1, 11, 2, 22], [4, 1]);
    const indices = tf.tensor1d([1, 0, 0, 1], 'int32');
    const dy = tf.tensor([23, 7, 19, 13], [4, 1]);
    const axis = 0;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(gradients, [26, 36, 0, 0]);
  });

  it('gradient 2D (gather) axis=1 shape=[2, 2]', () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    const indices = tf.tensor1d([1, 0, 0, 1], 'int32');
    const dy = tf.tensor([3, 4, 5, 6, 7, 8, 9, 10], [2, 4]);
    const axis = 1;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(gradients, [9, 9, 17, 17]);
  });

  it('gradient 2D (gather) axis=1 shape=[4, 1]', () => {
    const t = tf.tensor2d([1, 11, 2, 22], [4, 1]);
    const indices = tf.tensor1d([0, 0, 0, 0], 'int32');
    const dy = tf.tensor(
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], [4, 4]);
    const axis = 1;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(gradients, [18, 34, 50, 66]);
  });

  it('gradient 3D (gather) axis=0 shape=[2, 3, 2]', () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const indices = tf.tensor1d([1, 0, 0, 1], 'int32');
    const dy = tf.tensor(
        [
          2, -3, 4,  15, 6,  0.7, 1, 18, 0.01, 0,  12, 13,
          4, 15, 12, -7, 18, 19,  2, 21, 6,    23, 24, 25
        ],
        [4, 3, 2]);
    const axis = 0;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        gradients, [5, 33, 12.01, -7, 30, 32, 4, 18, 10, 38, 30, 25.7]);
  });

  it('gradient 3D (gather) axis=0 shape=[1, 4, 4]', () => {
    const t = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4]);
    const indices = tf.tensor1d([0, 0], 'int32');
    const dy = tf.tensor(
        [
          2,  -3, 4, 15, 6, 0.7, 1,  18, 0.01, 0,  12, 13, 4, 15, 12, -7,
          18, 19, 2, 21, 6, 23,  24, 25, 101,  31, 34, 54, 1, 0,  -3, -4
        ],
        [2, 4, 4]);
    const axis = 0;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        gradients,
        [20, 16, 6, 36, 12, 23.7, 25, 43, 101.01, 31, 46, 67, 5, 15, 9, -11]);
  });

  it('gradient 3D (gather) axis=1 shape=[2, 3, 2]', () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const indices = tf.tensor1d([1, 2, 2, 1], 'int32');
    const dy = tf.tensor(
        [2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 12, 13, 4, 15, 12, -7],
        [2, 4, 2]);
    const axis = 1;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        gradients, [0, 0, 3, 15, 10, 15.7, 0, 0, 12.01, -7, 16, 28]);
  });

  it('gradient 3D (gather) axis=1 shape=[1, 4, 4]', () => {
    const t = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4]);
    const indices = tf.tensor1d([1, 2, 2, 1], 'int32');
    const dy = tf.tensor(
        [2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 12, 13, 4, 15, 12, -7],
        [1, 4, 4]);
    const axis = 1;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        gradients, [0, 0, 0, 0, 6, 12, 16, 8, 6.01, .7, 13, 31, 0, 0, 0, 0]);
  });

  it('gradient 3D (gather) axis=2 shape=[2, 3, 2]', () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const indices = tf.tensor1d([1, 0, 1, 0], 'int32');
    const dy = tf.tensor(
        [
          2, -3, 4,  15, 6,  0.7, 1, 18, 0.01, 0,  12, 13,
          4, 15, 12, -7, 18, 19,  2, 21, 6,    23, 24, 25
        ],
        [2, 3, 4]);
    const axis = 2;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        gradients, [12, 6, 18.7, 7, 13, 12.01, 8, 16, 40, 20, 48, 30]);
  });

  it('gradient 3D (gather) axis=2 shape=[4, 1, 4]', () => {
    const t = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [4, 1, 4]);
    const indices = tf.tensor1d([1, 3, 1], 'int32');
    const dy =
        tf.tensor([2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 4, 15], [4, 1, 3]);
    const axis = 2;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        gradients, [0, 6, 0, -3, 0, 15.7, 0, 6, 0, 1.01, 0, 18, 0, 15, 0, 4]);
  });
});

describeWithFlags('oneHot', ALL_ENVS, () => {
  it('Depth 1 throws error', () => {
    const indices = tf.tensor1d([0, 0, 0], 'int32');
    expect(() => tf.oneHot(indices, 1)).toThrowError();
  });

  it('Depth 2, diagonal', () => {
    const indices = tf.tensor1d([0, 1], 'int32');
    const res = tf.oneHot(indices, 2);

    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(res, [1, 0, 0, 1]);
  });

  it('Depth 2, transposed diagonal', () => {
    const indices = tf.tensor1d([1, 0], 'int32');
    const res = tf.oneHot(indices, 2);

    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(res, [0, 1, 1, 0]);
  });

  it('Depth 3, 4 events', () => {
    const indices = tf.tensor1d([2, 1, 2, 0], 'int32');
    const res = tf.oneHot(indices, 3);

    expect(res.shape).toEqual([4, 3]);
    expectArraysClose(res, [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]);
  });

  it('Out of range events do not trigger onValue', () => {
    const indices = tf.tensor1d([-1, 5, 12345], 'int32');
    const res = tf.oneHot(indices, 5);
    expect(res.shape).toEqual([3, 5]);
    expectArraysClose(res, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('Depth 2 onValue=3, offValue=-2', () => {
    const indices = tf.tensor1d([0, 1], 'int32');
    const res = tf.oneHot(indices, 2, 3, -2);

    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(res, [3, -2, -2, 3]);
  });

  it('indices not int32 throws error', () => {
    const indices = tf.tensor1d([0, 1], 'float32');
    expect(() => tf.oneHot(indices, 2)).toThrowError();
  });

  it('check output dtype', () => {
    const expectedType = 'int32';
    const indices = tf.tensor1d([0, 1], 'int32');
    const res = tf.oneHot(indices, 2);

    expect(res.dtype).toEqual(expectedType);
  });
});

describeWithFlags('linspace', ALL_ENVS, () => {
  it('start stop', () => {
    const a = tf.linspace(1, 10, 10);
    expectArraysEqual(a, [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    expect(a.shape).toEqual([10]);

    const b = tf.linspace(12, 17, 8);
    expectArraysClose(b, [
      12., 12.71428571, 13.42857143, 14.14285714, 14.85714286, 15.57142857,
      16.28571429, 17.
    ]);
    expect(b.shape).toEqual([8]);

    const c = tf.linspace(9, 0, 6);
    expectArraysClose(c, [9., 7.2, 5.4, 3.6, 1.8, 0.]);
    expect(c.shape).toEqual([6]);
  });

  it('negative start stop', () => {
    const a = tf.linspace(-4, 5, 6);
    expectArraysClose(a, [-4., -2.2, -0.4, 1.4, 3.2, 5.]);
    expect(a.shape).toEqual([6]);
  });

  it('start negative stop', () => {
    const a = tf.linspace(4, -5, 6);
    expectArraysClose(a, [4., 2.2, 0.4, -1.4, -3.2, -5.]);
    expect(a.shape).toEqual([6]);
  });

  it('negative start negative stop', () => {
    const a = tf.linspace(-4, -5, 6);
    expectArraysClose(a, [-4., -4.2, -4.4, -4.6, -4.8, -5.]);
    expect(a.shape).toEqual([6]);

    const b = tf.linspace(-9, -4, 5);
    expectArraysClose(b, [-9., -7.75, -6.5, -5.25, -4.]);
    expect(b.shape).toEqual([5]);
  });

  it('should throw with no samples', () => {
    expect(() => tf.linspace(2, 10, 0)).toThrow();
  });
});

describeWithFlags('range', ALL_ENVS, () => {
  it('start stop', () => {
    const a = tf.range(0, 3);
    expectArraysEqual(a, [0, 1, 2]);
    expect(a.shape).toEqual([3]);

    const b = tf.range(3, 8);
    expectArraysEqual(b, [3, 4, 5, 6, 7]);
    expect(b.shape).toEqual([5]);
  });

  it('start stop negative', () => {
    const a = tf.range(-2, 3);
    expectArraysEqual(a, [-2, -1, 0, 1, 2]);
    expect(a.shape).toEqual([5]);

    const b = tf.range(4, -2);
    expectArraysEqual(b, [4, 3, 2, 1, 0, -1]);
    expect(b.shape).toEqual([6]);
  });

  it('start stop step', () => {
    const a = tf.range(4, 15, 4);
    expectArraysEqual(a, [4, 8, 12]);
    expect(a.shape).toEqual([3]);

    const b = tf.range(4, 11, 4);
    expectArraysEqual(b, [4, 8]);
    expect(b.shape).toEqual([2]);

    const c = tf.range(4, 17, 4);
    expectArraysEqual(c, [4, 8, 12, 16]);
    expect(c.shape).toEqual([4]);

    const d = tf.range(0, 30, 5);
    expectArraysEqual(d, [0, 5, 10, 15, 20, 25]);
    expect(d.shape).toEqual([6]);

    const e = tf.range(-3, 9, 2);
    expectArraysEqual(e, [-3, -1, 1, 3, 5, 7]);
    expect(e.shape).toEqual([6]);

    const f = tf.range(3, 3);
    expectArraysEqual(f, new Float32Array(0));
    expect(f.shape).toEqual([0]);

    const g = tf.range(3, 3, 1);
    expectArraysEqual(g, new Float32Array(0));
    expect(g.shape).toEqual([0]);

    const h = tf.range(3, 3, 4);
    expectArraysEqual(h, new Float32Array(0));
    expect(h.shape).toEqual([0]);

    const i = tf.range(-18, -2, 5);
    expectArraysEqual(i, [-18, -13, -8, -3]);
    expect(i.shape).toEqual([4]);
  });

  it('start stop large step', () => {
    const a = tf.range(3, 10, 150);
    expectArraysEqual(a, [3]);
    expect(a.shape).toEqual([1]);

    const b = tf.range(10, 500, 205);
    expectArraysEqual(b, [10, 215, 420]);
    expect(b.shape).toEqual([3]);

    const c = tf.range(3, -10, -150);
    expectArraysEqual(c, [3]);
    expect(c.shape).toEqual([1]);

    const d = tf.range(-10, -500, -205);
    expectArraysEqual(d, [-10, -215, -420]);
    expect(d.shape).toEqual([3]);
  });

  it('start stop negative step', () => {
    const a = tf.range(0, -10, -1);
    expectArraysEqual(a, [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]);
    expect(a.shape).toEqual([10]);

    const b = tf.range(0, -10);
    expectArraysEqual(b, [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]);
    expect(b.shape).toEqual([10]);

    const c = tf.range(3, -4, -2);
    expectArraysEqual(c, [3, 1, -1, -3]);
    expect(c.shape).toEqual([4]);

    const d = tf.range(-3, -18, -5);
    expectArraysEqual(d, [-3, -8, -13]);
    expect(d.shape).toEqual([3]);
  });

  it('start stop incompatible step', () => {
    const a = tf.range(3, 10, -2);
    expectArraysEqual(a, new Float32Array(0));
    expect(a.shape).toEqual([0]);

    const b = tf.range(40, 3, 2);
    expectArraysEqual(b, new Float32Array(0));
    expect(b.shape).toEqual([0]);
  });

  it('zero step', () => {
    expect(() => tf.range(2, 10, 0)).toThrow();
  });

  it('should have default dtype', () => {
    const a = tf.range(1, 4);
    expectArraysEqual(a, [1, 2, 3]);
    expect(a.dtype).toEqual('float32');
    expect(a.shape).toEqual([3]);
  });

  it('should have float32 dtype', () => {
    const a = tf.range(1, 4, undefined, 'float32');
    expectArraysEqual(a, [1, 2, 3]);
    expect(a.dtype).toEqual('float32');
    expect(a.shape).toEqual([3]);
  });

  it('should have int32 dtype', () => {
    const a = tf.range(1, 4, undefined, 'int32');
    expectArraysEqual(a, [1, 2, 3]);
    expect(a.dtype).toEqual('int32');
    expect(a.shape).toEqual([3]);
  });
});

describeWithFlags('fill', ALL_ENVS, () => {
  it('1D fill', () => {
    const a = tf.fill([3], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(a, [2, 2, 2]);
  });

  it('2D fill', () => {
    const a = tf.fill([3, 2], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(a, [2, 2, 2, 2, 2, 2]);
  });

  it('3D fill', () => {
    const a = tf.fill([3, 2, 1], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1]);
    expectArraysClose(a, [2, 2, 2, 2, 2, 2]);
  });

  it('4D fill', () => {
    const a = tf.fill([3, 2, 1, 2], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 2]);
    expectArraysClose(a, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
  });

  it('5D fill', () => {
    const a = tf.fill([2, 1, 2, 1, 2], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    expectArraysClose(a, [2, 2, 2, 2, 2, 2, 2, 2]);
  });
});

describeWithFlags('stack', ALL_ENVS, () => {
  it('scalars 3, 5 and 7', () => {
    const a = tf.scalar(3);
    const b = tf.scalar(5);
    const c = tf.scalar(7);
    const res = tf.stack([a, b, c]);
    expect(res.shape).toEqual([3]);
    expectArraysClose(res, [3, 5, 7]);
  });

  it('scalars 3, 5 and 7 along axis=1 throws error', () => {
    const a = tf.scalar(3);
    const b = tf.scalar(5);
    const c = tf.scalar(7);
    const f = () => tf.stack([a, b, c], 1);
    expect(f).toThrowError();
  });

  it('non matching shapes throws error', () => {
    const a = tf.scalar(3);
    const b = tf.tensor1d([5]);
    const f = () => tf.stack([a, b]);
    expect(f).toThrowError();
  });

  it('non matching dtypes throws error', () => {
    const a = tf.scalar(3);
    const b = tf.scalar(5, 'bool');
    const f = () => tf.stack([a, b]);
    expect(f).toThrowError();
  });

  it('2d but axis=3 throws error', () => {
    const a = tf.zeros([2, 2]);
    const b = tf.zeros([2, 2]);
    const f = () => tf.stack([a, b], 3 /* axis */);
    expect(f).toThrowError();
  });

  it('[1,2], [3,4] and [5,6], axis=0', () => {
    const a = tf.tensor1d([1, 2]);
    const b = tf.tensor1d([3, 4]);
    const c = tf.tensor1d([5, 6]);
    const res = tf.stack([a, b, c], 0 /* axis */);
    expect(res.shape).toEqual([3, 2]);
    expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
  });

  it('[1,2], [3,4] and [5,6], axis=1', () => {
    const a = tf.tensor1d([1, 2]);
    const b = tf.tensor1d([3, 4]);
    const c = tf.tensor1d([5, 6]);
    const res = tf.stack([a, b, c], 1 /* axis */);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(res, [1, 3, 5, 2, 4, 6]);
  });

  it('[[1,2],[3,4]] and [[5, 6], [7, 8]], axis=0', () => {
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const b = tf.tensor2d([[5, 6], [7, 8]]);
    const res = tf.stack([a, b], 0 /* axis */);
    expect(res.shape).toEqual([2, 2, 2]);
    expectArraysClose(res, [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('[[1,2],[3,4]] and [[5, 6], [7, 8]], axis=2', () => {
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const b = tf.tensor2d([[5, 6], [7, 8]]);
    const c = tf.tensor2d([[9, 10], [11, 12]]);
    const res = tf.stack([a, b, c], 2 /* axis */);
    expect(res.shape).toEqual([2, 2, 3]);
    expectArraysClose(res, [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]);
  });

  it('single tensor', () => {
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const res = tf.stack([a], 2 /* axis */);
    expect(res.shape).toEqual([2, 2, 1]);
    expectArraysClose(res, [1, 2, 3, 4]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.stack([{} as tf.Tensor]))
        .toThrowError(
            /Argument 'tensors\[0\]' passed to 'stack' must be a Tensor/);
  });
});

describeWithFlags('unstack', ALL_ENVS, () => {
  it('unstack by default', () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const res = tf.unstack(x);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(1);
    expect(res[0].shape).toEqual([4]);
    expectArraysClose(res[0], [1, 2, 3, 4]);
    expect(res[1].rank).toEqual(1);
    expect(res[1].shape).toEqual([4]);
    expectArraysClose(res[1], [5, 6, 7, 8]);
  });

  it('unstack into 3 tensors', () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    const res = tf.unstack(x, 0);
    expect(res.length).toEqual(3);
    expect(res[0].rank).toEqual(1);
    expect(res[0].shape).toEqual([2]);
    expectArraysClose(res[0], [1, 2]);
    expect(res[1].rank).toEqual(1);
    expect(res[1].shape).toEqual([2]);
    expectArraysClose(res[1], [3, 4]);
    expect(res[2].rank).toEqual(1);
    expect(res[2].shape).toEqual([2]);
    expectArraysClose(res[2], [5, 6]);
  });

  it('unstack by axis=1', () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const res = tf.unstack(x, 1);
    expect(res.length).toEqual(4);
    expect(res[0].rank).toEqual(1);
    expect(res[0].shape).toEqual([2]);
    expectArraysClose(res[0], [1, 5]);
    expect(res[1].rank).toEqual(1);
    expect(res[1].shape).toEqual([2]);
    expectArraysClose(res[1], [2, 6]);
    expect(res[2].rank).toEqual(1);
    expect(res[2].shape).toEqual([2]);
    expectArraysClose(res[2], [3, 7]);
    expect(res[3].rank).toEqual(1);
    expect(res[3].shape).toEqual([2]);
    expectArraysClose(res[3], [4, 8]);
  });

  it('unstack rank 3 tensor', () => {
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const res = tf.unstack(x);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(2);
    expect(res[0].shape).toEqual([2, 2]);
    expectArraysClose(res[0], [1, 2, 3, 4]);
    expect(res[1].rank).toEqual(2);
    expect(res[1].shape).toEqual([2, 2]);
    expectArraysClose(res[1], [5, 6, 7, 8]);
  });

  it('unstack rank 3 tensor with axis=1', () => {
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const res = tf.unstack(x, 1);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(2);
    expect(res[0].shape).toEqual([2, 2]);
    expectArraysClose(res[0], [1, 2, 5, 6]);
    expect(res[1].rank).toEqual(2);
    expect(res[1].shape).toEqual([2, 2]);
    expectArraysClose(res[1], [3, 4, 7, 8]);
  });

  it('unstack rank 3 tensor with axis=2', () => {
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const res = tf.unstack(x, 2);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(2);
    expect(res[0].shape).toEqual([2, 2]);
    expectArraysClose(res[0], [1, 3, 5, 7]);
    expect(res[1].rank).toEqual(2);
    expect(res[1].shape).toEqual([2, 2]);
    expectArraysClose(res[1], [2, 4, 6, 8]);
  });

  it('unstack rank 4 tensor', () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const res = tf.unstack(x);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(3);
    expect(res[0].shape).toEqual([2, 2, 1]);
    expectArraysClose(res[0], [1, 2, 3, 4]);
    expect(res[1].rank).toEqual(3);
    expect(res[1].shape).toEqual([2, 2, 1]);
    expectArraysClose(res[1], [5, 6, 7, 8]);
  });

  it('unstack rank 4 tensor with axis=1', () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const res = tf.unstack(x, 1);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(3);
    expect(res[0].shape).toEqual([2, 2, 1]);
    expectArraysClose(res[0], [1, 2, 5, 6]);
    expect(res[1].rank).toEqual(3);
    expect(res[1].shape).toEqual([2, 2, 1]);
    expectArraysClose(res[1], [3, 4, 7, 8]);
  });

  it('unstack rank 4 tensor with axis=2', () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const res = tf.unstack(x, 2);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(3);
    expect(res[0].shape).toEqual([2, 2, 1]);
    expectArraysClose(res[0], [1, 3, 5, 7]);
    expect(res[1].rank).toEqual(3);
    expect(res[1].shape).toEqual([2, 2, 1]);
    expectArraysClose(res[1], [2, 4, 6, 8]);
  });

  it('unstack rank 4 tensor with axis=3', () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const res = tf.unstack(x, 3);
    expect(res.length).toEqual(1);
    expect(res[0].rank).toEqual(3);
    expect(res[0].shape).toEqual([2, 2, 2]);
    expectArraysClose(res[0], [1, 2, 3, 4, 5, 6, 7, 8]);
  });
});

describeWithFlags('split', ALL_ENVS, () => {
  it('split by number', () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const res = tf.split(x, 2, 1);
    expect(res.length).toEqual(2);
    expect(res[0].shape).toEqual([2, 2]);
    expectArraysClose(res[0], [1, 2, 5, 6]);
    expect(res[1].shape).toEqual([2, 2]);
    expectArraysClose(res[1], [3, 4, 7, 8]);
  });

  it('split by sizes', () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const res = tf.split(x, [1, 2, 1], 1);
    expect(res.length).toEqual(3);
    expect(res[0].shape).toEqual([2, 1]);
    expectArraysClose(res[0], [1, 5]);
    expect(res[1].shape).toEqual([2, 2]);
    expectArraysClose(res[1], [2, 3, 6, 7]);
    expect(res[2].shape).toEqual([2, 1]);
    expectArraysClose(res[2], [4, 8]);
  });

  it('sizes to not sum to axis size throws error', () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const f = () => tf.split(x, [1, 2], 1);
    expect(f).toThrowError();
  });

  it('number of splits does not evenly divide axis', () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const f = () => tf.split(x, 3, 1);
    expect(f).toThrowError();
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.split({} as tf.Tensor, 1))
        .toThrowError(/Argument 'x' passed to 'split' must be a Tensor/);
  });
});

describeWithFlags('expandDims', ALL_ENVS, () => {
  it('scalar, default axis is 0', () => {
    const res = tf.scalar(1).expandDims();
    expect(res.shape).toEqual([1]);
    expectArraysClose(res, [1]);
  });

  it('scalar, axis is out of bounds throws error', () => {
    const f = () => tf.scalar(1).expandDims(1);
    expect(f).toThrowError();
  });

  it('1d, axis=0', () => {
    const res = tf.tensor1d([1, 2, 3]).expandDims(0 /* axis */);
    expect(res.shape).toEqual([1, 3]);
    expectArraysClose(res, [1, 2, 3]);
  });

  it('1d, axis=1', () => {
    const res = tf.tensor1d([1, 2, 3]).expandDims(1 /* axis */);
    expect(res.shape).toEqual([3, 1]);
    expectArraysClose(res, [1, 2, 3]);
  });

  it('2d, axis=0', () => {
    const res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(0 /* axis */);
    expect(res.shape).toEqual([1, 3, 2]);
    expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
  });

  it('2d, axis=1', () => {
    const res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(1 /* axis */);
    expect(res.shape).toEqual([3, 1, 2]);
    expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
  });

  it('2d, axis=2', () => {
    const res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(2 /* axis */);
    expect(res.shape).toEqual([3, 2, 1]);
    expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
  });

  it('4d, axis=0', () => {
    const res = tf.tensor4d([[[[4]]]]).expandDims();
    expect(res.shape).toEqual([1, 1, 1, 1, 1]);
    expectArraysClose(res, [4]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.expandDims({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'expandDims' must be a Tensor/);
  });
});

describeWithFlags('cumsum', ALL_ENVS, () => {
  it('1D standard', () => {
    const res = tf.tensor1d([1, 2, 3, 4]).cumsum();
    expect(res.shape).toEqual([4]);
    expectArraysClose(res, [1, 3, 6, 10]);
  });

  it('1D reverse', () => {
    const reverse = true;
    const exclusive = false;
    const res = tf.tensor1d([1, 2, 3, 4]).cumsum(0, exclusive, reverse);
    expect(res.shape).toEqual([4]);
    expectArraysClose(res, [10, 9, 7, 4]);
  });

  it('1D exclusive', () => {
    const exclusive = true;
    const res = tf.tensor1d([1, 2, 3, 4]).cumsum(0, exclusive);
    expect(res.shape).toEqual([4]);
    expectArraysClose(res, [0, 1, 3, 6]);
  });

  it('1D exclusive reverse', () => {
    const reverse = true;
    const exclusive = true;
    const res = tf.tensor1d([1, 2, 3, 4]).cumsum(0, exclusive, reverse);
    expect(res.shape).toEqual([4]);
    expectArraysClose(res, [9, 7, 4, 0]);
  });

  it('gradient: 1D', () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([4, 5, 6]);
    const da = tf.grad(x => tf.cumsum(x))(a, dy);

    expect(da.shape).toEqual([3]);
    expectArraysClose(da, [15, 11, 6]);
  });

  it('2D standard', () => {
    const res = tf.tensor2d([[1, 2], [3, 4]]).cumsum(1);
    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(res, [1, 3, 3, 7]);
  });

  it('2D reverse exclusive', () => {
    const reverse = true;
    const exclusive = true;
    const res = tf.tensor2d([[1, 2], [3, 4]]).cumsum(1, exclusive, reverse);
    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(res, [2, 0, 4, 0]);
  });

  it('2D axis=0', () => {
    const res = tf.tensor2d([[1, 2], [3, 4]]).cumsum();
    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(res, [1, 2, 4, 6]);
  });

  it('3D standard', () => {
    const res = tf.tensor3d([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]).cumsum(2);
    expect(res.shape).toEqual([2, 2, 2]);
    expectArraysClose(res, [0, 1, 2, 5, 4, 9, 6, 13]);
  });
});
