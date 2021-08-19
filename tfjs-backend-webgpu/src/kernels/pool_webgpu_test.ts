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
import {test_util} from '@tensorflow/tfjs-core';
import {describeWebGPU} from '../test_util';

const expectArraysClose = test_util.expectArraysClose;

describeWebGPU('pool', () => {
  // For PoolWithFilterSizeEqualsOneProgram. This case will fail wasm, so keep
  // it here. Wasm bug: https://github.com/tensorflow/tfjs/issues/5471.
  // TODO(xing.xu): https://github.com/tensorflow/tfjs/issues/5506.
  it('x=[4,4,1] f=[1,1] s=2 d=1', async () => {
    // Feed forward.
    const a = tf.tensor3d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [4, 4, 1]);

    const windowShape = 1;
    const padding = 0;
    const dilationRate: number = undefined;
    const strides = 2;

    const result =
        tf.pool(a, windowShape, 'avg', padding, dilationRate, strides);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [0, 2, 8, 10]);
  });
});
