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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

describeWithFlags('floorDiv', ALL_ENVS, () => {
  it('floorDiv', async () => {
    const a = tf.tensor1d([10, 20, -20, -40], 'int32');
    const b = tf.tensor1d([10, 12, 8, 5], 'int32');
    const result = tf.floorDiv(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [1, 1, -3, -8]);
  });

  it('floorDiv vec4', async () => {
    const a = tf.tensor1d([10, 20, -20, -40], 'int32');
    const b = tf.tensor1d([10, 12, 8, 5], 'int32');
    const result = tf.floorDiv(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [1, 1, -3, -8]);
  });
});
