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

describeWithFlags('conv3dTranspose', ALL_ENVS, () => {
  // Reference Python TensorFlow code
  // ```python
  // import numpy as np
  // import tensorflow as tf
  // tf.enable_eager_execution()
  // x = np.array([2], dtype = np.float32).reshape(1, 1, 1, 1, 1)
  // w = np.array([5, 4, 8, 7, 1, 2, 6, 3], dtype = np.float32).reshape(2, 2, 2,
  //   1, 1)
  // tf.nn.conv3d_transpose(x, w, output_shape=[1, 2, 2, 2, 1], padding='VALID')
  // ```
  it('input=2x2x2x1,d2=1,f=2,s=1,p=valid', async () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const inputShape: [number, number, number, number] =
        [1, 1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 'valid';
    const origStride = 1;

    const x = tf.tensor4d([2], inputShape);
    const w = tf.tensor5d(
        [5, 4, 8, 7, 1, 2, 6, 3],
        [fSize, fSize, fSize, origInputDepth, origOutputDepth]);

    const result = tf.conv3dTranspose(x, w, [2, 2, 2, 1], origStride, origPad);
    const expected = [10, 8, 16, 14, 2, 4, 12, 6];

    expect(result.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  // Reference Python TensorFlow code
  // ```python
  // import numpy as np
  // import tensorflow as tf
  // tf.enable_eager_execution()
  // x = np.array([2, 3], dtype = np.float32).reshape(2, 1, 1, 1, 1, 1)
  // w = np.array([5, 4, 8, 7, 1, 2, 6, 3], dtype = np.float32).reshape(2,
  //   2, 2, 1, 1)
  // tf.nn.conv3d_transpose(x, w, output_shape=[2, 2, 2, 2, 1], padding='VALID')
  // ```
  it('input=2x2x2x1,d2=1,f=2,s=1,p=valid, batch=2', async () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const inputShape: [number, number, number, number, number] =
        [2, 1, 1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 'valid';
    const origStride = 1;

    const x = tf.tensor5d([2, 3], inputShape);
    const w = tf.tensor5d(
        [5, 4, 8, 7, 1, 2, 6, 3],
        [fSize, fSize, fSize, origInputDepth, origOutputDepth]);

    const result =
        tf.conv3dTranspose(x, w, [2, 2, 2, 2, 1], origStride, origPad);
    const expected = [10, 8, 16, 14, 2, 4, 12, 6, 15, 12, 24, 21, 3, 6, 18, 9];

    expect(result.shape).toEqual([2, 2, 2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });
});
