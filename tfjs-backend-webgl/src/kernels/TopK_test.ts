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

import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {WEBGL_ENVS} from '../backend_webgl_test_registry';

describeWithFlags('TopK', WEBGL_ENVS, () => {
  it('handles packed inputs', async () => {
    const a = tf.tensor1d([
      1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2,
      1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2,
      1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1
    ]);

    // pack a using the add op which packs outputs
    tf.env().set('WEBGL_PACK', true);
    const aPacked = tf.addN([a, tf.zeros(a.shape)]);

    const k = a.shape[0];
    const {values, indices} = tf.topk(aPacked, k);

    expect(values.shape).toEqual([k]);
    expect(indices.shape).toEqual([k]);
    expect(values.dtype).toBe('float32');
    expect(indices.dtype).toBe('int32');

    tf.test_util.expectArraysEqual(await values.data(), [
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]);
    tf.test_util.expectArraysEqual(await indices.data(), [
      2,  4,  8,  9,  12, 15, 18, 21, 23, 27, 30, 33, 38, 40, 41, 42, 43,
      45, 47, 48, 49, 51, 52, 54, 55, 57, 61, 63, 0,  1,  3,  5,  6,  7,
      10, 11, 13, 14, 16, 17, 19, 20, 22, 24, 25, 26, 28, 29, 31, 32, 34,
      35, 36, 37, 39, 44, 46, 50, 53, 56, 58, 59, 60, 62, 64
    ]);
  });
});
