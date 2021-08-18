/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

describeWithFlags('spaceToBatchND', ALL_ENVS, () => {
  it('tensor4d, input shape=[1, 2, 2, 1], blockShape=[2, 2]', async () => {
    const t = tf.tensor4d([[[[1], [2]], [[3], [4]]]], [1, 2, 2, 1]);
    const blockShape = [2, 2];
    const paddings = [[0, 0], [0, 0]];

    const res = tf.spaceToBatchND(t, blockShape, paddings);
    expect(res.shape).toEqual([4, 1, 1, 1]);
    expectArraysClose(await res.data(), [1, 2, 3, 4]);
  });

  it('tensor4d, input shape=[1, 2, 2, 3], blockShape=[2, 2]', async () => {
    const t = tf.tensor4d(
        [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], [1, 2, 2, 3]);
    const blockShape = [2, 2];
    const paddings = [[0, 0], [0, 0]];

    const res = tf.spaceToBatchND(t, blockShape, paddings);
    expect(res.shape).toEqual([4, 1, 1, 3]);
    expectArraysClose(
        await res.data(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  });

  it('tensor4d, input shape=[1, 4, 4, 1], blockShape=[2, 2]', async () => {
    const t = tf.tensor4d(
        [[
          [[1], [2], [3], [4]], [[5], [6], [7], [8]], [[9], [10], [11], [12]],
          [[13], [14], [15], [16]]
        ]],
        [1, 4, 4, 1]);
    const blockShape = [2, 2];
    const paddings = [[0, 0], [0, 0]];

    const res = tf.spaceToBatchND(t, blockShape, paddings);
    expect(res.shape).toEqual([4, 2, 2, 1]);
    expectArraysClose(
        await res.data(),
        [1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16]);
  });

  it('tensor4d, input shape=[2, 6, 6, 1], blockShape=[2, 2]', async () => {
    const t = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
          46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
          61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
        ],
        [2, 6, 6, 1]);
    const blockShape = [2, 2];
    const paddings = [[0, 0], [0, 0]];

    const res = tf.spaceToBatchND(t, blockShape, paddings);
    expect(res.shape).toEqual([8, 3, 3, 1]);
    expectArraysClose(await res.data(), [
      1, 3,  5,  13, 15, 17, 25, 27, 29, 37, 39, 41, 49, 51, 53, 61, 63, 65,
      2, 4,  6,  14, 16, 18, 26, 28, 30, 38, 40, 42, 50, 52, 54, 62, 64, 66,
      7, 9,  11, 19, 21, 23, 31, 33, 35, 43, 45, 47, 55, 57, 59, 67, 69, 71,
      8, 10, 12, 20, 22, 24, 32, 34, 36, 44, 46, 48, 56, 58, 60, 68, 70, 72
    ]);
  });

  it('tensor4d, input shape=[2, 2, 4, 1], blockShape=[2, 2]', async () => {
    const t = tf.tensor4d(
        [
          [[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
          [[[9], [10], [11], [12]], [[13], [14], [15], [16]]]
        ],
        [2, 2, 4, 1]);
    const blockShape = [2, 2];
    const paddings = [[0, 0], [2, 0]];

    const res = tf.spaceToBatchND(t, blockShape, paddings);
    expect(res.shape).toEqual([8, 1, 3, 1]);
    expectArraysClose(await res.data(), [
      0, 1, 3, 0, 9,  11, 0, 2, 4, 0, 10, 12,
      0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16
    ]);
  });

  it('tensor2d, blockShape [2]', async () => {
    const t = tf.tensor2d([1, 3, 2, 4], [1, 4]);
    const blockShape = [2];
    const paddings = [[0, 0]];

    const res = tf.spaceToBatchND(t, blockShape, paddings);
    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(await res.data(), [1, 2, 3, 4]);
  });

  it('throws when blockShape equal to input rank', () => {
    const t = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const blockShape = [2, 2, 2, 2];
    const paddings = [[0, 0], [0, 0], [0, 0], [0, 0]];

    expect(() => tf.spaceToBatchND(t, blockShape, paddings))
        .toThrowError('input rank 4 should be > than [blockShape] 4');
  });

  it('throws when paddings row dimension not equal to blockshape', () => {
    const t = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const blockShape = [2, 2];
    const paddings = [[0, 0]];

    expect(() => tf.spaceToBatchND(t, blockShape, paddings))
        .toThrowError('paddings.shape[0] 1 must be equal to [blockShape] 2');
  });

  it('throws when input tensor spatial dimension not divisible by blockshapes',
     () => {
       const t = tf.tensor4d([1, 2, 3, 4, 5, 6], [1, 2, 3, 1]);
       const blockShape = [2, 2];
       const paddings = [[0, 0], [0, 0]];

       expect(() => tf.spaceToBatchND(t, blockShape, paddings))
           .toThrowError(
               'input spatial dimensions 2,3,1 with paddings 0,0,0,0 must be ' +
               'divisible by blockShapes 2,2');
     });

  it('accepts a tensor-like object', async () => {
    const t = [[[[1], [2]], [[3], [4]]]];
    const blockShape = [2, 2];
    const paddings = [[0, 0], [0, 0]];

    const res = tf.spaceToBatchND(t, blockShape, paddings);
    expect(res.shape).toEqual([4, 1, 1, 1]);
    expectArraysClose(await res.data(), [1, 2, 3, 4]);
  });
});

describeWithFlags('batchToSpaceND X spaceToBatchND', ALL_ENVS, () => {
  it('tensor4d, input shape=[4, 1, 1, 1], blockShape=[2, 2]', async () => {
    const t = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
    const blockShape = [2, 2];
    const crops = [[0, 0], [0, 0]];
    const paddings = [[0, 0], [0, 0]];

    const b2s = tf.batchToSpaceND(t, blockShape, crops);
    expect(b2s.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await b2s.data(), [1, 2, 3, 4]);

    const s2b = tf.spaceToBatchND(b2s, blockShape, paddings);
    expect(s2b.shape).toEqual([4, 1, 1, 1]);
    expectArraysClose(await s2b.data(), [1, 2, 3, 4]);
  });

  it('tensor4d, input shape=[2, 6, 6, 1], blockShape=[2, 2]', async () => {
    const t = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
          46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
          61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
        ],
        [2, 6, 6, 1]);
    const blockShape = [2, 2];
    const crops = [[0, 0], [0, 0]];
    const paddings = [[0, 0], [0, 0]];

    const s2b = tf.spaceToBatchND(t, blockShape, paddings);
    expect(s2b.shape).toEqual([8, 3, 3, 1]);
    expectArraysClose(await s2b.data(), [
      1, 3,  5,  13, 15, 17, 25, 27, 29, 37, 39, 41, 49, 51, 53, 61, 63, 65,
      2, 4,  6,  14, 16, 18, 26, 28, 30, 38, 40, 42, 50, 52, 54, 62, 64, 66,
      7, 9,  11, 19, 21, 23, 31, 33, 35, 43, 45, 47, 55, 57, 59, 67, 69, 71,
      8, 10, 12, 20, 22, 24, 32, 34, 36, 44, 46, 48, 56, 58, 60, 68, 70, 72
    ]);

    const b2s = tf.batchToSpaceND(s2b, blockShape, crops);
    expect(b2s.shape).toEqual([2, 6, 6, 1]);
    expectArraysClose(await b2s.data(), [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
      19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
      37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
      55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
    ]);
  });

  it('gradients,  input shape=[4, 2, 2], block shape=[2]', async () => {
    const t = tf.tensor(
        [-61, 37, -68, 72, 31, 62, 0, -13, 28, 54, 96, 44, -55, -64, -88, -94],
        [4, 2, 2]);
    const blockShape = [2];
    const paddings = [[0, 2]];
    const dy = tf.tensor(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ],
        [8, 2, 2]);

    const gradient =
        tf.grad(t => tf.spaceToBatchND(t, blockShape, paddings))(t, dy);
    expect(gradient.shape).toEqual([4, 2, 2]);
    expectArraysClose(
        await gradient.data(),
        [1, 2, 17, 18, 5, 6, 21, 22, 9, 10, 25, 26, 13, 14, 29, 30]);
  });

  it('gradient with clones input=[4, 2, 2], block shape=[2]', async () => {
    const t = tf.tensor(
        [-61, 37, -68, 72, 31, 62, 0, -13, 28, 54, 96, 44, -55, -64, -88, -94],
        [4, 2, 2]);
    const blockShape = [2];
    const paddings = [[0, 2]];
    const dy = tf.tensor(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ],
        [8, 2, 2]);

    const gradient = tf.grad(
        t => tf.spaceToBatchND(t.clone(), blockShape, paddings).clone())(t, dy);
    expect(gradient.shape).toEqual([4, 2, 2]);
    expectArraysClose(
        await gradient.data(),
        [1, 2, 17, 18, 5, 6, 21, 22, 9, 10, 25, 26, 13, 14, 29, 30]);
  });

  it('gradients, input shape=[2, 2, 4, 1], block shape=[2, 2]', async () => {
    const t = tf.tensor4d(
        [
          [[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
          [[[9], [10], [11], [12]], [[13], [14], [15], [16]]]
        ],
        [2, 2, 4, 1]);
    const blockShape = [2, 2];
    const paddings = [[0, 0], [2, 0]];
    const dy = tf.tensor(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ],
        [8, 1, 3, 1]);

    const gradient =
        tf.grad(t => tf.spaceToBatchND(t, blockShape, paddings))(t, dy);
    expect(gradient.shape).toEqual([2, 2, 4, 1]);
    expectArraysClose(
        await gradient.data(),
        [2, 8, 3, 9, 14, 20, 15, 21, 5, 11, 6, 12, 17, 23, 18, 24]);
  });
});
