/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {BackendWasm} from './index';

/**
 * Tests specific to the wasm backend. The name of these tests must start with
 * 'wasm' so that they are always included in the test runner. See
 * `env.specFilter` in `setup_test.ts` for details.
 */
describeWithFlags('wasm', ALL_ENVS, () => {
  it('write and read values', async () => {
    const x = tf.tensor1d([1, 2, 3]);
    test_util.expectArraysClose([1, 2, 3], await x.data());
  });

  it('allocate repetitively and confirm reuse of heap space', () => {
    const backend = tf.backend() as BackendWasm;
    const size = 100;
    // Allocate for the first time, record the memory offset and dispose.
    const t1 = tf.zeros([size]);
    const memOffset1 = backend.getMemoryOffset(t1.dataId);
    t1.dispose();

    // Allocate again and make sure the offset is the same (memory was reused).
    const t2 = tf.zeros([size]);
    const memOffset2 = backend.getMemoryOffset(t2.dataId);
    // This should fail in case of a memory leak.
    expect(memOffset1).toBe(memOffset2);
  });
});
