/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

const {expectArraysClose} = test_util;

describeWithFlags('Conv2D WebGL Implementation ', ALL_ENVS, () => {
  it('should work when width is odd and called multiple times.', async () => {
    const filter = tf.tensor4d([-1, 3, 2, 1, 3, 4, 4, -2], [1, 1, 4, 2]);
    const image = tf.tensor3d(
        [
          111, 112, 113, 114, 121, 122, 123, 124, 131, 132, 133, 134,
          211, 212, 213, 214, 221, 222, 223, 224, 231, 232, 233, 234,
          311, 312, 313, 314, 321, 322, 323, 324, 331, 332, 333, 334,

        ],
        [3, 3, 4]);

    tf.conv2d(image, filter, 1, 'valid');
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = tf.conv2d(image, filter, 1, 'valid');
    const resultData = await result.data();

    const expected = [
      908, 669, 988, 729, 1068, 789, 1708, 1269, 1788, 1329, 1868, 1389, 2508,
      1869, 2588, 1929, 2668, 1989
    ];

    expectArraysClose(resultData, expected);
  });

  it('image is packed and isChannelFirst.', async () => {
    const filter = tf.tensor4d([1], [1, 1, 1, 1]);
    const image = tf.tensor3d([11, 12, 13, 21, 22, 23, 31, 32, 33], [1, 3, 3]);

    // pack image.
    tf.mul(image, 1);

    tf.conv2d(image, filter, 1, 'valid', 'NCHW');
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = tf.conv2d(image, filter, 1, 'valid', 'NCHW');
    const resultData = await result.data();

    const expected = [11, 12, 13, 21, 22, 23, 31, 32, 33];

    expectArraysClose(resultData, expected);
  });

  it('image is unpacked and isChannelFirst.', async () => {
    const filter = tf.tensor4d([1], [1, 1, 1, 1]);
    const image = tf.tensor3d([11, 12, 13, 21, 22, 23, 31, 32, 33], [1, 3, 3]);

    tf.conv2d(image, filter, 1, 'valid', 'NCHW');
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = tf.conv2d(image, filter, 1, 'valid', 'NCHW');
    const resultData = await result.data();

    const expected = [11, 12, 13, 21, 22, 23, 31, 32, 33];

    expectArraysClose(resultData, expected);
  });

  it('image is packed and isChannelLast.', async () => {
    const filter = tf.tensor4d([1], [1, 1, 1, 1]);
    const image = tf.tensor3d([11, 12, 13, 21, 22, 23, 31, 32, 33], [3, 3, 1]);

    // pack image.
    tf.mul(image, 1);

    tf.conv2d(image, filter, 1, 'valid');
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = tf.conv2d(image, filter, 1, 'valid');
    const resultData = await result.data();

    const expected = [11, 12, 13, 21, 22, 23, 31, 32, 33];

    expectArraysClose(resultData, expected);
  });

  it('image is unpacked and isChannelLast.', async () => {
    const filter = tf.tensor4d([1], [1, 1, 1, 1]);
    const image = tf.tensor3d([11, 12, 13, 21, 22, 23, 31, 32, 33], [3, 3, 1]);

    tf.conv2d(image, filter, 1, 'valid');
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = tf.conv2d(image, filter, 1, 'valid');
    const resultData = await result.data();

    const expected = [11, 12, 13, 21, 22, 23, 31, 32, 33];

    expectArraysClose(resultData, expected);
  });
});
