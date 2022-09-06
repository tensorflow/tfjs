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

import {WebGLMemoryInfo} from './backend_webgl';
import {WEBGL_ENVS} from './backend_webgl_test_registry';

describeWithFlags('toPixels', WEBGL_ENVS, () => {
  it('does not leak memory', async () => {
    const x = tf.tensor2d([[.1], [.2]], [2, 1]);
    const startNumBytesInGPU = (tf.memory() as WebGLMemoryInfo).numBytesInGPU;
    await tf.browser.toPixels(x);
    expect((tf.memory() as WebGLMemoryInfo).numBytesInGPU)
        .toEqual(startNumBytesInGPU);
  });

  it('does not leak memory given a tensor-like object', async () => {
    const x = [[10], [20]];  // 2x1;
    const startNumBytesInGPU = (tf.memory() as WebGLMemoryInfo).numBytesInGPU;
    await tf.browser.toPixels(x);
    expect((tf.memory() as WebGLMemoryInfo).numBytesInGPU)
        .toEqual(startNumBytesInGPU);
  });
});
