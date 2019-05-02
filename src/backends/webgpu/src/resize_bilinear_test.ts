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

import * as tf from '@tensorflow/tfjs-core';
import * as tfwebgpu from './index';

describe('resizeBilinear', () => {
  beforeAll(async () => tfwebgpu.ready);

  it('simple alignCorners=false', async () => {
    const input = tf.tensor3d([2, 2, 4, 4], [2, 2, 1]);
    const output = input.resizeBilinear([3, 3], false);

    tf.test_util.expectArraysClose(
        await output.data(),
        new Float32Array([2, 2, 2, 10 / 3, 10 / 3, 10 / 3, 4, 4, 4]));
  });

  it('simple alignCorners=true', async () => {
    const input = tf.tensor3d([2, 2, 4, 4], [2, 2, 1]);
    const output = input.resizeBilinear([3, 3], true);

    tf.test_util.expectArraysClose(
        await output.data(), new Float32Array([2, 2, 2, 3, 3, 3, 4, 4, 4]));
  });

  it('matches tensorflow w/ random numbers alignCorners=false', async () => {
    const input = tf.tensor3d(
        [
          1.19074044, 0.91373104, 2.01611669, -0.52270832, 0.38725395,
          1.30809779, 0.61835143, 3.49600659, 2.09230986, 0.56473997,
          0.03823943, 1.19864896
        ],
        [2, 3, 2]);
    const output = input.resizeBilinear([4, 5], false);

    tf.test_util.expectArraysClose(
        await output.data(), new Float32Array([
          1.19074047,  0.91373104, 1.68596613, 0.05186744, 1.69034398,
          -0.15654698, 0.7130264,  0.94193673, 0.38725394, 1.30809784,
          0.9045459,   2.20486879, 1.59434628, 0.89455694, 1.68591988,
          0.26748738,  0.58103991, 1.00690198, 0.21274668, 1.25337338,
          0.6183514,   3.49600649, 1.50272655, 1.73724651, 1.68149579,
          0.69152176,  0.44905344, 1.07186723, 0.03823943, 1.19864893,
          0.6183514,   3.49600649, 1.50272655, 1.73724651, 1.68149579,
          0.69152176,  0.44905344, 1.07186723, 0.03823943, 1.19864893
        ]));
  });

  it('matches tensorflow w/ random numbers alignCorners=true', async () => {
    const input = tf.tensor3d(
        [
          1.56324531, 2.13817752, 1.44398421, 1.07632684, 0.59306785,
          -0.36970865, 1.62451879, 1.8367334, 1.13944798, 2.01993218,
          2.01919952, 2.67524054
        ],
        [2, 3, 2]);
    const output = input.resizeBilinear([4, 5], true);

    tf.test_util.expectArraysClose(
        await output.data(), new Float32Array([
          1.5632453,  2.13817763, 1.50361478, 1.60725224, 1.44398427,
          1.07632685, 1.01852608, 0.35330909, 0.59306782, -0.36970866,
          1.58366978, 2.03769612, 1.46307099, 1.71427906, 1.3424722,
          1.39086199, 1.20545864, 1.01806819, 1.06844509, 0.6452744,
          1.60409427, 1.93721485, 1.42252707, 1.82130599, 1.24096,
          1.70539713, 1.3923912,  1.68282723, 1.54382229, 1.66025746,
          1.62451875, 1.83673346, 1.38198328, 1.92833281, 1.13944793,
          2.01993227, 1.57932377, 2.34758639, 2.01919961, 2.67524052
        ]));
  });

  it('batch of 2, simple, alignCorners=true', async () => {
    const input = tf.tensor4d([2, 2, 4, 4, 3, 3, 5, 5], [2, 2, 2, 1]);
    const output = input.resizeBilinear([3, 3], true /* alignCorners */);

    tf.test_util.expectArraysClose(
        await output.data(),
        new Float32Array(
            [2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4, 5, 5, 5]));
  });

  it('target width = 1, alignCorners=true', async () => {
    const input = tf.tensor3d([
      [
        [120.68856811523438, 134.51638793945312, 83.03671264648438],
        [121.58008575439453, 113.28836059570312, 136.3172149658203],
        [79.38370513916016, 101.87127685546875, 104.54979705810547],
        [96.31678771972656, 111.77168273925781, 83.73509979248047]
      ],
      [
        [119.45088195800781, 88.98846435546875, 97.47553253173828],
        [117.5562973022461, 108.26356506347656, 99.62212371826172],
        [136.62701416015625, 94.10433197021484, 80.97366333007812],
        [83.61205291748047, 90.60148620605469, 81.82512664794922]
      ],
      [
        [103.0362777709961, 123.1098403930664, 125.62944030761719],
        [92.2915267944336, 103.15729522705078, 119.18060302734375],
        [102.93293762207031, 117.821044921875, 99.40152740478516],
        [96.32952117919922, 105.80963134765625, 104.8491439819336]
      ],
      [
        [104.87507629394531, 134.0189208984375, 111.02627563476562],
        [85.4534683227539, 107.68426513671875, 103.03722381591797],
        [89.70533752441406, 98.25298309326172, 78.42916870117188],
        [113.6744613647461, 95.8189697265625, 122.75005340576172]
      ]
    ]);

    const output = input.resizeBilinear([3, 1], true);

    const expected = new Float32Array([
      120.68857, 134.51639, 83.03671, 111.243576, 106.04915, 111.55249,
      104.87508, 134.01892, 111.026276
    ]);
    tf.test_util.expectArraysClose(await output.data(), expected);
    expect(output.shape).toEqual([3, 1, 3]);
  });

  it('target height = 1, alignCorners=true', async () => {
    const input = tf.tensor3d([
      [
        [120.68856811523438, 134.51638793945312, 83.03671264648438],
        [121.58008575439453, 113.28836059570312, 136.3172149658203],
        [79.38370513916016, 101.87127685546875, 104.54979705810547],
        [96.31678771972656, 111.77168273925781, 83.73509979248047]
      ],
      [
        [119.45088195800781, 88.98846435546875, 97.47553253173828],
        [117.5562973022461, 108.26356506347656, 99.62212371826172],
        [136.62701416015625, 94.10433197021484, 80.97366333007812],
        [83.61205291748047, 90.60148620605469, 81.82512664794922]
      ],
      [
        [103.0362777709961, 123.1098403930664, 125.62944030761719],
        [92.2915267944336, 103.15729522705078, 119.18060302734375],
        [102.93293762207031, 117.821044921875, 99.40152740478516],
        [96.32952117919922, 105.80963134765625, 104.8491439819336]
      ],
      [
        [104.87507629394531, 134.0189208984375, 111.02627563476562],
        [85.4534683227539, 107.68426513671875, 103.03722381591797],
        [89.70533752441406, 98.25298309326172, 78.42916870117188],
        [113.6744613647461, 95.8189697265625, 122.75005340576172]
      ]
    ]);

    const output = input.resizeBilinear([1, 3], true);

    const expected = new Float32Array([
      120.68857, 134.51639, 83.03671, 100.481895, 107.57982, 120.4335, 96.31679,
      111.77168, 83.7351
    ]);
    tf.test_util.expectArraysClose(await output.data(), expected);
    expect(output.shape).toEqual([1, 3, 3]);
  });

  it('throws when passed a non-tensor', () => {
    const e = /Argument 'images' passed to 'resizeBilinear' must be a Tensor/;
    expect(() => tf.image.resizeBilinear({} as tf.Tensor3D, [
      1, 1
    ])).toThrowError(e);
  });

  it('accepts a tensor-like object', async () => {
    const input = [[[2], [2]], [[4], [4]]];  // 2x2x1
    const output = tf.image.resizeBilinear(input, [3, 3], false);
    tf.test_util.expectArraysClose(
        await output.data(),
        new Float32Array([2, 2, 2, 10 / 3, 10 / 3, 10 / 3, 4, 4, 4]));
  });
});
