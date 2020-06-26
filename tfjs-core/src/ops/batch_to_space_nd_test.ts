/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

describeWithFlags('batchToSpaceND', ALL_ENVS, () => {
  it('tensor4d, input shape=[4, 1, 1, 1], blockShape=[2, 2]', async () => {
    const t = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
    const blockShape = [2, 2];
    const crops = [[0, 0], [0, 0]];

    const res = tf.batchToSpaceND(t, blockShape, crops);
    expect(res.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await res.data(), [1, 2, 3, 4]);
  });

  it('tensor4d, input shape=[4, 1, 1, 3], blockShape=[2, 2]', async () => {
    const t =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 1, 1, 3]);
    const blockShape = [2, 2];
    const crops = [[0, 0], [0, 0]];

    const res = tf.batchToSpaceND(t, blockShape, crops);
    expect(res.shape).toEqual([1, 2, 2, 3]);
    expectArraysClose(
        await res.data(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  });

  it('tensor4d, input shape=[4, 2, 2, 1], blockShape=[2, 2]', async () => {
    const t = tf.tensor4d(
        [1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16], [4, 2, 2, 1]);
    const blockShape = [2, 2];
    const crops = [[0, 0], [0, 0]];

    const res = tf.batchToSpaceND(t, blockShape, crops);
    expect(res.shape).toEqual([1, 4, 4, 1]);
    expectArraysClose(
        await res.data(),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
  });

  it('tensor4d, input shape=[8, 1, 3, 1], blockShape=[2, 2]', async () => {
    const t = tf.tensor4d(
        [
          0, 1, 3, 0, 9,  11, 0, 2, 4, 0, 10, 12,
          0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16
        ],
        [8, 1, 3, 1]);
    const blockShape = [2, 2];
    const crops = [[0, 0], [2, 0]];

    const res = tf.batchToSpaceND(t, blockShape, crops);
    expect(res.shape).toEqual([2, 2, 4, 1]);
    expectArraysClose(
        await res.data(),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
  });

  it('tensor2d, blockShape [1]', async () => {
    const t = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const blockShape = [2];
    const crops = [[0, 0]];

    const res = tf.batchToSpaceND(t, blockShape, crops);
    expect(res.shape).toEqual([1, 4]);
    expectArraysClose(await res.data(), [1, 3, 2, 4]);
  });

  it('tensor3d,  blockSHape [1]', async () => {
    const t = tf.tensor(
        [
          -61, 37,  -68, 72,  31,  62, 0,   -13, 28,  54, 96,
          44,  -55, -64, -88, -94, 65, -32, -96, -73, -2, -77,
          -14, 47,  33,  15,  70,  20, 75,  28,  84,  -13
        ],
        [8, 2, 2]);
    const blockShape = [2];
    const crops = [[0, 2]];

    const res = tf.batchToSpaceND(t, blockShape, crops);
    expect(res.shape).toEqual([4, 2, 2]);
    expectArraysClose(
        await res.data(),
        [-61, 37, 65, -32, 31, 62, -2, -77, 28, 54, 33, 15, -55, -64, 75, 28]);
  });

  it('tensor3d, blockShape [2]', async () => {
    const t = tf.tensor(
        [
          -61, 37,  -68, 72,  31,  62, 0,   -13, 28,  54, 96,
          44,  -55, -64, -88, -94, 65, -32, -96, -73, -2, -77,
          -14, 47,  33,  15,  70,  20, 75,  28,  84,  -13
        ],
        [8, 2, 2]);
    const blockShape = [2, 2];
    const crops = [[2, 0], [2, 0]];

    const res = tf.batchToSpaceND(t, blockShape, crops);
    expect(res.shape).toEqual([2, 2, 2]);
    expectArraysClose(await res.data(), [72, 44, -73, 20, -13, -94, 47, -13]);
  });

  it('throws when blockShape equal to input rank', () => {
    const t = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
    const blockShape = [2, 2, 2, 2];
    const crops = [[0, 0], [0, 0], [0, 0], [0, 0]];

    expect(() => tf.batchToSpaceND(t, blockShape, crops))
        .toThrowError(
            `input rank is ${t.rank} but should be > than blockShape.length ${
                blockShape.length}`);
  });

  it('throws when crops row dimension not equal to blockshape', () => {
    const t = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
    const blockShape = [2, 2];
    const crops = [[0, 0]];

    expect(() => tf.batchToSpaceND(t, blockShape, crops))
        .toThrowError(`crops.length is ${
            crops.length} but should be equal to blockShape.length  ${
            blockShape.length}`);
  });

  it('throws when input tensor batch not divisible by prod(blockShape)', () => {
    const t = tf.tensor4d([1, 2, 3, 4, 5], [5, 1, 1, 1]);
    const blockShape = [2, 2];
    const crops = [[0, 0], [0, 0]];
    const prod = blockShape.reduce((a, b) => a * b);

    expect(() => tf.batchToSpaceND(t, blockShape, crops))
        .toThrowError(
            `input tensor batch is ${t.shape[0]} but is not divisible by the ` +
            `product of the elements of blockShape ${
                blockShape.join(' * ')} === ${prod}`);
  });

  it('accepts a tensor-like object', async () => {
    const t = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]];
    const blockShape = [2, 2];
    const crops = [[0, 0], [0, 0]];

    const res = tf.batchToSpaceND(t, blockShape, crops);
    expect(res.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await res.data(), [1, 2, 3, 4]);
  });

  it('gradients,  input shape=[4, 2, 2], block shape=[2]', async () => {
    const t = tf.tensor(
        [-61, 37, -68, 72, 31, 62, 0, -13, 28, 54, 96, 44, -55, -64, -88, -94],
        [4, 2, 2]);
    const blockShape = [2];
    const crops = [[0, 2]];
    const dy = tf.tensor([.01, .02, .03, .04, .05, .06, .07, .08], [2, 2, 2]);

    const gradient =
        tf.grad(t => tf.batchToSpaceND(t, blockShape, crops))(t, dy);
    expect(gradient.shape).toEqual([4, 2, 2]);
    expectArraysClose(await gradient.data(), [
      0.01, 0.02, 0, 0, 0.05, 0.06, 0, 0, 0.03, 0.04, 0, 0, 0.07, 0.08, 0, 0
    ]);
  });

  it('gradients, input shape=[4, 2, 2, 1], block shape=[2, 2]', async () => {
    const t = tf.tensor4d(
        [1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16], [4, 2, 2, 1]);
    const blockShape = [2, 2];
    const crops = [[0, 0], [0, 0]];
    const dy = tf.tensor(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4, 1]);

    const gradient =
        tf.grad(t => tf.batchToSpaceND(t, blockShape, crops))(t, dy);
    expect(gradient.shape).toEqual([4, 2, 2, 1]);
    expectArraysClose(
        await gradient.data(),
        [1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16]);
  });

  it('gradient with clones, input=[4, 2, 2, 1], block shape=[2, 2]',
     async () => {
       const t = tf.tensor4d(
           [1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16],
           [4, 2, 2, 1]);
       const blockShape = [2, 2];
       const crops = [[0, 0], [0, 0]];
       const dy = tf.tensor(
           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
           [1, 4, 4, 1]);

       const gradient = tf.grad(
           t => tf.batchToSpaceND(t.clone(), blockShape, crops).clone())(t, dy);
       expect(gradient.shape).toEqual([4, 2, 2, 1]);
       expectArraysClose(
           await gradient.data(),
           [1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16]);
     });
});
