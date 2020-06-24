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

describeWithFlags('relu6', ALL_ENVS, () => {
  it('basic relu6', async () => {
    const a = tf.tensor1d([1, -2, 0, 8, -0.1]);
    const result = tf.relu6(a);
    expectArraysClose(await result.data(), [1, 0, 0, 6, 0]);
  });

  it('gradients: relu6', async () => {
    const a = tf.scalar(8);
    const dy = tf.scalar(5);

    const grad = tf.grad(a => tf.relu6(a));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [0]);
  });

  it('gradients: relu6 array', async () => {
    const a = tf.tensor2d([8, -1, 0, .1], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grad = tf.grad(a => tf.relu6(a));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [0, 0, 0, 4]);
  });
});
