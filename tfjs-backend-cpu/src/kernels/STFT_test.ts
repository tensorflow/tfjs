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
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

describeWithFlags('stft memory test', ALL_ENVS, () => {
  it('should have no mem leak', async () => {
    const win = 320;
    const fft = 320;
    const hop = 160;
    const input = tf.zeros<tf.Rank.R1>([1760]);

    const startTensors = tf.memory().numTensors;
    const startDataIds = tf.engine().backend.numDataIds();
    const result = await tf.signal.stft(input, win, hop, fft);

    // 1 new tensor, 3 new data buckets.
    expect(tf.memory().numTensors).toBe(startTensors + 1);
    expect(tf.engine().backend.numDataIds()).toBe(startTensors + 3);

    result.dispose();

    // Zero net tensors / data buckets.
    expect(tf.memory().numTensors).toBe(startTensors);
    expect(tf.engine().backend.numDataIds()).toBe(startDataIds);
    input.dispose();
  });
});
