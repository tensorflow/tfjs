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

describeWithFlags('reverse', ALL_ENVS, () => {
  it('throws when passed a non-tensor', () => {
    expect(() => tf.reverse({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'reverse' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const input = [1, 2, 3];
    const result = tf.reverse(input);
    expect(result.shape).toEqual([3]);
    expectArraysClose(await result.data(), [3, 2, 1]);
  });

  it('ensure no memory leak', async () => {
    const numTensorsBefore = tf.memory().numTensors;
    const numDataIdBefore = tf.engine().backend.numDataIds();

    const input = tf.tensor1d([1, 2, 3]);
    const result = tf.reverse(input);
    expect(result.shape).toEqual([3]);
    expectArraysClose(await result.data(), [3, 2, 1]);

    input.dispose();
    result.dispose();

    const numTensorsAfter = tf.memory().numTensors;
    const numDataIdAfter = tf.engine().backend.numDataIds();
    expect(numTensorsAfter).toBe(numTensorsBefore);
    expect(numDataIdAfter).toBe(numDataIdBefore);
  });
});
