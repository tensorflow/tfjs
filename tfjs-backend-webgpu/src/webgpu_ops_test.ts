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

import {test_util} from '@tensorflow/tfjs-core';
const expectArraysClose = test_util.expectArraysClose;
import * as tf from '@tensorflow/tfjs-core';

import {describeWebGPU} from './test_util';

describeWebGPU('gather', () => {
  it('fills with zero when index is out of bound', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    const tInt = tf.tensor2d([1, 11, 2, 22], [2, 2], 'int32');

    const index = tf.tensor1d([0, 1, 100, -1, 2, -4], 'int32');
    const res = tf.gather(t, index);
    const resInt = tf.gather(tInt, index);

    const expected = [1, 11, 2, 22, 0, 0, 0, 0, 0, 0, 0, 0];
    expectArraysClose(await res.data(), expected);
    expectArraysClose(await resInt.data(), expected);
  });
});
