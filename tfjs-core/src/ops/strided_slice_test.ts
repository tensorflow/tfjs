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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

describeWithFlags('stridedSlice', ALL_ENVS, () => {
  it('with ellipsisMask=1', async () => {
    const t = tf.tensor2d([
      [1, 2, 3, 4, 5],
      [2, 3, 4, 5, 6],
      [3, 4, 5, 6, 7],
      [4, 5, 6, 7, 8],
      [5, 6, 7, 8, 9],
      [6, 7, 8, 9, 10],
      [7, 8, 9, 10, 11],
      [8, 8, 9, 10, 11],
      [9, 8, 9, 10, 11],
      [10, 8, 9, 10, 11],
    ]);
    const begin = [0, 4];
    const end = [0, 5];
    const strides = [1, 1];
    const beginMask = 0;
    const endMask = 0;
    const ellipsisMask = 1;
    const output =
        t.stridedSlice(begin, end, strides, beginMask, endMask, ellipsisMask);
    expect(output.shape).toEqual([10, 1]);
    expectArraysClose(await output.data(), [5, 6, 7, 8, 9, 10, 11, 11, 11, 11]);
  });

  it('with ellipsisMask=1, begin / end masks and start / end normalization',
     async () => {
       const t = tf.randomNormal([1, 6, 2006, 4]);
       const output =
           tf.stridedSlice(t, [0, 0, 0], [0, 2004, 0], [1, 1, 1], 6, 4, 1);
       expect(output.shape).toEqual([1, 6, 2004, 4]);
     });

  it('with ellipsisMask=1 and start / end normalization', async () => {
    const t = tf.tensor3d([
      [[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]
    ]);
    const begin = [1, 0];
    const end = [2, 1];
    const strides = [1, 1];
    const beginMask = 0;
    const endMask = 0;
    const ellipsisMask = 1;

    const output = tf.stridedSlice(
        t, begin, end, strides, beginMask, endMask, ellipsisMask);
    expect(output.shape).toEqual([3, 2, 1]);
    expectArraysClose(await output.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('with ellipsisMask=2', async () => {
    const t = tf.tensor3d([
      [[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]
    ]);
    const begin = [1, 0, 0];
    const end = [2, 1, 3];
    const strides = [1, 1, 1];
    const beginMask = 0;
    const endMask = 0;
    const ellipsisMask = 2;
    const output = tf.stridedSlice(
        t, begin, end, strides, beginMask, endMask, ellipsisMask);
    expect(output.shape).toEqual([1, 2, 3]);
    expectArraysClose(await output.data(), [3, 3, 3, 4, 4, 4]);
  });

  it('with ellipsisMask=2 and start / end normalization', async () => {
    const t = tf.tensor4d([
      [[[1, 1], [1, 1], [1, 1]], [[2, 2], [2, 2], [2, 2]]],

      [[[3, 3], [3, 3], [3, 3]], [[4, 4], [4, 4], [4, 4]]],

      [[[5, 5], [5, 5], [5, 5]], [[6, 6], [6, 6], [6, 6]]]
    ]);

    const begin = [1, 0, 0];
    const end = [2, 1, 1];
    const strides = [1, 1, 1];
    const beginMask = 0;
    const endMask = 0;
    const ellipsisMask = 2;
    const output = tf.stridedSlice(
        t, begin, end, strides, beginMask, endMask, ellipsisMask);
    expect(output.shape).toEqual([1, 2, 3, 1]);
    expectArraysClose(await output.data(), [3, 3, 3, 4, 4, 4]);
  });

  it('stridedSlice should fail if ellipsis mask is set and newAxisMask or ' +
         'shrinkAxisMask are also set',
     async () => {
       const tensor = tf.tensor1d([0, 1, 2, 3]);
       expect(() => tf.stridedSlice(tensor, [0], [3], [2], 0, 0, 1, 1))
           .toThrow();
       expect(() => tf.stridedSlice(tensor, [0], [3], [2], 0, 0, 1, 0, 1))
           .toThrow();
     });

  it('stridedSlice with first axis being new', async () => {
    // Python slice code: t[tf.newaxis,0:3]
    const t = tf.tensor1d([0, 1, 2, 3]);
    const begin = [0, 0];
    const end = [1, 3];
    const strides = [1, 2];
    const beginMask = 0;
    const endMask = 0;
    const ellipsisMask = 0;
    const newAxisMask = 1;

    const output = tf.stridedSlice(
        t, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask);
    expect(output.shape).toEqual([1, 2]);
    expectArraysClose(await output.data(), [0, 2]);
  });

  it('strided slice with several new axes', async () => {
    // Python slice code: t[1:2,tf.newaxis,0:3,tf.newaxis,2:5]
    const t = tf.zeros([2, 3, 4, 5]);
    const begin = [1, 0, 0, 0, 2];
    const end = [2, 1, 3, 1, 5];
    const strides: number[] = null;
    const beginMask = 0;
    const endMask = 0;
    const ellipsisMask = 0;
    const newAxisMask = 0b1010;
    const output = tf.stridedSlice(
        t, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask);
    expect(output.shape).toEqual([1, 1, 3, 1, 2, 5]);
    expectArraysClose(await output.data(), new Array(30).fill(0));
  });

  it('strided slice with new axes and shrink axes', () => {
    // Python slice code: t[1:2,tf.newaxis,1,tf.newaxis,2,2:5]
    const t = tf.zeros([2, 3, 4, 5]);
    const begin = [1, 0, 1, 0, 2, 2];
    const end = [2, 1, 2, 1, 3, 5];
    const strides: number[] = null;
    const beginMask = 0;
    const endMask = 0;
    const ellipsisMask = 0;
    const newAxisMask = 0b1010;
    const shrinkAxisMask = 0b10100;
    const output = tf.stridedSlice(
        t, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask,
        shrinkAxisMask);
    expect(output.shape).toEqual([1, 1, 1, 3]);
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

  it('stridedSlice should handle negative end with ellipsisMask', () => {
    const a = tf.ones([1, 240, 1, 10]);
    const output =
        tf.stridedSlice(a, [0, 0, 0], [0, -1, 0], [1, 1, 1], 3, 1, 4);
    expect(output.shape).toEqual([1, 239, 1, 10]);
  });
  it('accepts a tensor-like object', async () => {
    const tensor = [0, 1, 2, 3];
    const output = tf.stridedSlice(tensor, [0], [3], [2]);
    expect(output.shape).toEqual([2]);
    expectArraysClose(await output.data(), [0, 2]);
  });

  it('ensure no memory leak', async () => {
    const numTensorsBefore = tf.memory().numTensors;
    const numDataIdBefore = tf.engine().backend.numDataIds();

    const tensor = tf.tensor1d([0, 1, 2, 3]);
    const output = tf.stridedSlice(tensor, [0], [3], [2]);
    expect(output.shape).toEqual([2]);
    expectArraysClose(await output.data(), [0, 2]);

    tensor.dispose();
    output.dispose();

    const numTensorsAfter = tf.memory().numTensors;
    const numDataIdAfter = tf.engine().backend.numDataIds();
    expect(numTensorsAfter).toBe(numTensorsBefore);
    expect(numDataIdAfter).toBe(numDataIdBefore);
  });
});
