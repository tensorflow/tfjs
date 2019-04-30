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
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

describeWithFlags('stridedSlice', ALL_ENVS, () => {
  it('stridedSlice should fail if new axis mask is set', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    expect(() => tf.stridedSlice(tensor, [0], [3], [2], 0, 0, 0, 1)).toThrow();
  });

  it('stridedSlice should fail if ellipsis mask is set', () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    expect(() => tf.stridedSlice(tensor, [0], [3], [2], 0, 0, 1)).toThrow();
  });

  it('stridedSlice should support 1d tensor', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [0], [3], [2]);
    expect(output.shape).toEqual([2]);
    expectArraysClose(await output.data(), [0, 2]);
  });

  it('stridedSlice should support 1d tensor', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [0], [3], [2]);
    expect(output.shape).toEqual([2]);
    expectArraysClose(await output.data(), [0, 2]);
  });

  it('stridedSlice with 1d tensor should be used by tensor directly',
     async () => {
       const t = tf.tensor1d([0, 1, 2, 3]);
       const output = t.stridedSlice([0], [3], [2]);
       expect(output.shape).toEqual([2]);
       expectArraysClose(await output.data(), [0, 2]);
     });

  it('stridedSlice should support 1d tensor empty result', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [10], [3], [2]);
    expect(output.shape).toEqual([0]);
    expectArraysClose(await output.data(), []);
  });

  it('stridedSlice should support 1d tensor negative begin', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [-3], [3], [1]);
    expect(output.shape).toEqual([2]);
    expectArraysClose(await output.data(), [1, 2]);
  });

  it('stridedSlice should support 1d tensor out of range begin', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [-5], [3], [1]);
    expect(output.shape).toEqual([3]);
    expectArraysClose(await output.data(), [0, 1, 2]);
  });

  it('stridedSlice should support 1d tensor negative end', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [1], [-2], [1]);
    expect(output.shape).toEqual([1]);
    expectArraysClose(await output.data(), [1]);
  });

  it('stridedSlice should support 1d tensor out of range end', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [-3], [5], [1]);
    expect(output.shape).toEqual([3]);
    expectArraysClose(await output.data(), [1, 2, 3]);
  });

  it('stridedSlice should support 1d tensor begin mask', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [1], [3], [1], 1);
    expect(output.shape).toEqual([3]);
    expectArraysClose(await output.data(), [0, 1, 2]);
  });

  it('stridedSlice should support 1d tensor nagtive begin and stride',
     async () => {
       const tensor = tf.tensor1d([0, 1, 2, 3]);
       const output = tf.stridedSlice(tensor, [-2], [-3], [-1]);
       expect(output.shape).toEqual([1]);
       expectArraysClose(await output.data(), [2]);
     });

  it('stridedSlice should support 1d tensor' +
         ' out of range begin and negative stride',
     async () => {
       const tensor = tf.tensor1d([0, 1, 2, 3]);
       const output = tf.stridedSlice(tensor, [5], [-2], [-1]);
       expect(output.shape).toEqual([1]);
       expectArraysClose(await output.data(), [3]);
     });

  it('stridedSlice should support 1d tensor nagtive end and stride',
     async () => {
       const tensor = tf.tensor1d([0, 1, 2, 3]);
       const output = tf.stridedSlice(tensor, [2], [-4], [-1]);
       expect(output.shape).toEqual([2]);
       expectArraysClose(await output.data(), [2, 1]);
     });

  it('stridedSlice should support 1d tensor' +
         ' out of range end and negative stride',
     async () => {
       const tensor = tf.tensor1d([0, 1, 2, 3]);
       const output = tf.stridedSlice(tensor, [-3], [-5], [-1]);
       expect(output.shape).toEqual([2]);
       expectArraysClose(await output.data(), [1, 0]);
     });

  it('stridedSlice should support 1d tensor end mask', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [1], [3], [1], 0, 1);
    expect(output.shape).toEqual([3]);
    expectArraysClose(await output.data(), [1, 2, 3]);
  });

  it('stridedSlice should support 1d tensor shrink axis mask', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [1], [3], [1], 0, 0, 0, 0, 1);
    expect(output.shape).toEqual([]);
    expectArraysClose(await output.data(), [1]);
  });

  it('stridedSlice should support 1d tensor negative stride', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [-1], [-4], [-1]);
    expect(output.shape).toEqual([3]);
    expectArraysClose(await output.data(), [3, 2, 1]);
  });

  it('stridedSlice should support 1d tensor even length stride', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [0], [2], [2]);
    expect(output.shape).toEqual([1]);
    expectArraysClose(await output.data(), [0]);
  });

  it('stridedSlice should support 1d tensor odd length stride', async () => {
    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [0], [3], [2]);
    expect(output.shape).toEqual([2]);
    expectArraysClose(await output.data(), [0, 2]);
  });

  it('stridedSlice should support 2d tensor identity', async () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output = tf.stridedSlice(tensor, [0, 0], [2, 3], [1, 1]);
    expect(output.shape).toEqual([2, 3]);
    expectArraysClose(await output.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('stridedSlice should support 2d tensor', async () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output = tf.stridedSlice(tensor, [1, 0], [2, 2], [1, 1]);
    expect(output.shape).toEqual([1, 2]);
    expectArraysClose(await output.data(), [4, 5]);
  });

  it('stridedSlice should support 2d tensor strides', async () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output = tf.stridedSlice(tensor, [0, 0], [2, 3], [2, 2]);
    expect(output.shape).toEqual([1, 2]);
    expectArraysClose(await output.data(), [1, 3]);
  });

  it('stridedSlice with 2d tensor should be used by tensor directly',
     async () => {
       const t = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
       const output = t.stridedSlice([1, 0], [2, 2], [1, 1]);
       expect(output.shape).toEqual([1, 2]);
       expectArraysClose(await output.data(), [4, 5]);
     });

  it('stridedSlice should support 2d tensor negative strides', async () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output = tf.stridedSlice(tensor, [1, -1], [2, -4], [2, -1]);
    expect(output.shape).toEqual([1, 3]);
    expectArraysClose(await output.data(), [6, 5, 4]);
  });

  it('stridedSlice should support 2d tensor begin mask', async () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output = tf.stridedSlice(tensor, [1, 0], [2, 2], [1, 1], 1);
    expect(output.shape).toEqual([2, 2]);
    expectArraysClose(await output.data(), [1, 2, 4, 5]);
  });

  it('stridedSlice should support 2d tensor shrink mask', async () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output =
        tf.stridedSlice(tensor, [1, 0], [2, 2], [1, 1], 0, 0, 0, 0, 1);
    expect(output.shape).toEqual([2]);
    expectArraysClose(await output.data(), [4, 5]);
  });

  it('stridedSlice should support 2d tensor end mask', async () => {
    const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const output = tf.stridedSlice(tensor, [1, 0], [2, 2], [1, 1], 0, 2);
    expect(output.shape).toEqual([1, 3]);
    expectArraysClose(await output.data(), [4, 5, 6]);
  });

  it('stridedSlice should support 2d tensor' +
         ' negative strides and begin mask',
     async () => {
       const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
       const output = tf.stridedSlice(tensor, [1, -2], [2, -4], [1, -1], 2);
       expect(output.shape).toEqual([1, 3]);
       expectArraysClose(await output.data(), [6, 5, 4]);
     });

  it('stridedSlice should support 2d tensor' +
         ' negative strides and end mask',
     async () => {
       const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
       const output = tf.stridedSlice(tensor, [1, -2], [2, -3], [1, -1], 0, 2);
       expect(output.shape).toEqual([1, 2]);
       expectArraysClose(await output.data(), [5, 4]);
     });

  it('stridedSlice should support 3d tensor identity', async () => {
    const tensor =
        tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const output = tf.stridedSlice(tensor, [0, 0, 0], [2, 3, 2], [1, 1, 1]);
    expect(output.shape).toEqual([2, 3, 2]);
    expectArraysClose(
        await output.data(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  });

  it('stridedSlice should support 3d tensor negative stride', async () => {
    const tensor =
        tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const output =
        tf.stridedSlice(tensor, [-1, -1, -1], [-3, -4, -3], [-1, -1, -1]);
    expect(output.shape).toEqual([2, 3, 2]);
    expectArraysClose(
        await output.data(), [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
  });

  it('stridedSlice should support 3d tensor strided 2', async () => {
    const tensor =
        tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const output = tf.stridedSlice(tensor, [0, 0, 0], [2, 3, 2], [2, 2, 2]);
    expect(output.shape).toEqual([1, 2, 1]);
    expectArraysClose(await output.data(), [1, 5]);
  });

  it('stridedSlice should support 3d tensor shrink mask', async () => {
    const tensor =
        tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const output =
        tf.stridedSlice(tensor, [0, 0, 0], [2, 3, 2], [1, 1, 1], 0, 0, 0, 0, 1);
    expect(output.shape).toEqual([3, 2]);
    expectArraysClose(await output.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('stridedSlice should support 3d with smaller length of begin array',
     async () => {
       const tensor =
           tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 1, 2]);
       const output = tf.stridedSlice(
           tensor, [1, 0], [2, 3, 1, 2], [1, 1, 1, 1], 0, 0, 0, 0, 0);
       expect(output.shape).toEqual([1, 3, 1, 2]);
       expectArraysClose(await output.data(), [7, 8, 9, 10, 11, 12]);
     });

  it('stridedSlice should support 3d with smaller length of end array',
     async () => {
       const tensor =
           tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 1, 2]);
       const output = tf.stridedSlice(
           tensor, [1, 0, 0, 0], [2, 3], [1, 1, 1, 1], 0, 0, 0, 0, 0);
       expect(output.shape).toEqual([1, 3, 1, 2]);
       expectArraysClose(await output.data(), [7, 8, 9, 10, 11, 12]);
     });

  it('stridedSlice should support 3d with smaller length of stride array',
     async () => {
       const tensor =
           tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 1, 2]);
       const output = tf.stridedSlice(
           tensor, [1, 0, 0, 0], [2, 3, 1, 2], [1, 1], 0, 0, 0, 0, 0);
       expect(output.shape).toEqual([1, 3, 1, 2]);
       expectArraysClose(await output.data(), [7, 8, 9, 10, 11, 12]);
     });

  it('stridedSlice should throw when passed a non-tensor', () => {
    expect(() => tf.stridedSlice({} as tf.Tensor, [0], [0], [1]))
        .toThrowError(/Argument 'x' passed to 'stridedSlice' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const tensor = [0, 1, 2, 3];
    const output = tf.stridedSlice(tensor, [0], [3], [2]);
    expect(output.shape).toEqual([2]);
    expectArraysClose(await output.data(), [0, 2]);
  });
});
