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

import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
import {ALL_ENVS, expectArraysClose} from '../test_util';

describeWithFlags('stridedSlice', ALL_ENVS, () => {
  it('stridedSlice should suport 1d tensor', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [0], [3], [2]);
    expect(output.shape).toEqual([2]);
    expectArraysClose(output, [0, 2]);
  });

  it('stridedSlice should suport 1d tensor empty result', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [10], [3], [2]);
    expect(output.shape).toEqual([0]);
    expectArraysClose(output, []);
  });

  it('stridedSlice should suport 1d tensor negative begin', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [-3], [3], [1]);
    expect(output.shape).toEqual([2]);
    expectArraysClose(output, [1, 2]);
  });

  it('stridedSlice should suport 1d tensor out of range begin', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [-5], [3], [1]);
    expect(output.shape).toEqual([3]);
    expectArraysClose(output, [0, 1, 2]);
  });

  it('stridedSlice should suport 1d tensor negative end', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [1], [-2], [1]);
    expect(output.shape).toEqual([1]);
    expectArraysClose(output, [1]);
  });

  it('stridedSlice should suport 1d tensor out of range end', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [-3], [5], [1]);
    expect(output.shape).toEqual([3]);
    expectArraysClose(output, [1, 2, 3]);
  });

  it('stridedSlice should suport 1d tensor begin mask', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [1], [3], [1], 1);
    expect(output.shape).toEqual([3]);
    expectArraysClose(output, [0, 1, 2]);
  });

  it('stridedSlice should suport 1d tensor nagtive begin and stride', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [-2], [-3], [-1]);
    expect(output.shape).toEqual([1]);
    expectArraysClose(output, [2]);
  });

  it('stridedSlice should suport 1d tensor' +
         ' out of range begin and negative stride',
     () => {
       const tensor = tf.tensor1d([0, 1, 2, 3]);
       const output = tf.stridedSlice(tensor, [5], [-2], [-1]);
       expect(output.shape).toEqual([1]);
       expectArraysClose(output, [3]);
     });

  it('stridedSlice should suport 1d tensor nagtive end and stride', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [2], [-4], [-1]);
    expect(output.shape).toEqual([2]);
    expectArraysClose(output, [2, 1]);
  });

  it('stridedSlice should suport 1d tensor' +
         ' out of range end and negative stride',
     () => {
       const tensor = tf.tensor1d([0, 1, 2, 3]);
       const output = tf.stridedSlice(tensor, [-3], [-5], [-1]);
       expect(output.shape).toEqual([2]);
       expectArraysClose(output, [1, 0]);
     });

  it('stridedSlice should suport 1d tensor end mask', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [1], [3], [1], 0, 1);
    expect(output.shape).toEqual([3]);
    expectArraysClose(output, [1, 2, 3]);
  });

  it('stridedSlice should suport 1d tensor negative stride', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [-1], [-4], [-1]);
    expect(output.shape).toEqual([3]);
    expectArraysClose(output, [3, 2, 1]);
  });

  it('stridedSlice should suport 1d tensor even length stride', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [0], [2], [2]);
    expect(output.shape).toEqual([1]);
    expectArraysClose(output, [0]);
  });

  it('stridedSlice should suport 1d tensor odd length stride', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [0], [3], [2]);
    expect(output.shape).toEqual([2]);
    expectArraysClose(output, [0, 2]);
  });

  it('stridedSlice should suport 2d tensor identity', () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output = tf.stridedSlice(tensor, [0, 0], [2, 3], [1, 1]);
    expect(output.shape).toEqual([2, 3]);
    expectArraysClose(output, [1, 2, 3, 4, 5, 6]);
  });

  it('stridedSlice should suport 2d tensor', () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output = tf.stridedSlice(tensor, [1, 0], [2, 2], [1, 1]);
    expect(output.shape).toEqual([1, 2]);
    expectArraysClose(output, [4, 5]);
  });

  it('stridedSlice should suport 2d tensor strides', () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output = tf.stridedSlice(tensor, [0, 0], [2, 3], [2, 2]);
    expect(output.shape).toEqual([1, 2]);
    expectArraysClose(output, [1, 3]);
  });

  it('stridedSlice should suport 2d tensor negative strides', () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output = tf.stridedSlice(tensor, [1, -1], [2, -4], [2, -1]);
    expect(output.shape).toEqual([1, 3]);
    expectArraysClose(output, [6, 5, 4]);
  });

  it('stridedSlice should suport 2d tensor begin mask', () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output = tf.stridedSlice(tensor, [1, 0], [2, 2], [1, 1], 1);
    expect(output.shape).toEqual([2, 2]);
    expectArraysClose(output, [1, 2, 4, 5]);
  });

  it('stridedSlice should suport 2d tensor end mask', () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output = tf.stridedSlice(tensor, [1, 0], [2, 2], [1, 1], 0, 2);
    expect(output.shape).toEqual([1, 3]);
    expectArraysClose(output, [4, 5, 6]);
  });

  it('stridedSlice should suport 2d tensor' +
         ' negative strides and begin mask',
     () => {
       const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
       const output = tf.stridedSlice(tensor, [1, -2], [2, -4], [1, -1], 2);
       expect(output.shape).toEqual([1, 3]);
       expectArraysClose(output, [6, 5, 4]);
     });

  it('stridedSlice should suport 2d tensor' +
         ' negative strides and end mask',
     () => {
       const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
       const output = tf.stridedSlice(tensor, [1, -2], [2, -3], [1, -1], 0, 2);
       expect(output.shape).toEqual([1, 2]);
       expectArraysClose(output, [5, 4]);
     });

  it('stridedSlice should suport 3d tensor identity', () => {
    const tensor =
        tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const output = tf.stridedSlice(tensor, [0, 0, 0], [2, 3, 2], [1, 1, 1]);
    expect(output.shape).toEqual([2, 3, 2]);
    expectArraysClose(output, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  });

  it('stridedSlice should suport 3d tensor negative stride', () => {
    const tensor =
        tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const output =
        tf.stridedSlice(tensor, [-1, -1, -1], [-3, -4, -3], [-1, -1, -1]);
    expect(output.shape).toEqual([2, 3, 2]);
    expectArraysClose(output, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
  });

  it('stridedSlice should suport 3d tensor strided 2', () => {
    const tensor =
        tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const output = tf.stridedSlice(tensor, [0, 0, 0], [2, 3, 2], [2, 2, 2]);
    expect(output.shape).toEqual([1, 2, 1]);
    expectArraysClose(output, [1, 5]);
  });

  it('stridedSlice should throw when passed a non-tensor', () => {
    expect(() => tf.stridedSlice({} as tf.Tensor, [0], [0], [1]))
        .toThrowError(/Argument 'x' passed to 'stridedSlice' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const tensor = [0, 1, 2, 3];
    const output = tf.stridedSlice(tensor, [0], [3], [2]);
    expect(output.shape).toEqual([2]);
    expectArraysClose(output, [0, 2]);
  });
});
