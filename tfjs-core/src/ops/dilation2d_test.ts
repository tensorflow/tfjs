/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

describeWithFlags('dilation2d', ALL_ENVS, () => {
  it('Should throw error.', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const filterShape: [number, number, number] = [1, 1, inputDepth];

    const pad = 'same';
    const stride: [number, number] = [1, 1];

    const x = tf.tensor3d([1, 1, 1, 1], inputShape);
    const filter = tf.tensor3d([1], filterShape);

    const result = tf.dilation2d(x, filter, stride, pad);

    expectArraysClose(await result.data(), [1, 1, 1, 1]);
  });
});
