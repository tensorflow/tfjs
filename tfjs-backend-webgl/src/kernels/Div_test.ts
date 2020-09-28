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
import {test_util} from '@tensorflow/tfjs-core';
const {expectArraysClose} = test_util;
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as webgl_util from '../webgl_util';

describeWithFlags('Div.', ALL_ENVS, () => {
  fit('does not have memory leak.', async () => {
    const MAX_FLOAT16 = 65504;
    spyOn(webgl_util, 'canBeRepresented').and.callFake((val: number) => {
      if (val > MAX_FLOAT16) {
        return false;
      }
      return true;
    });

    // We can't flip the WEBGL_RENDER_FLOAT32_ENABLED flag to test this because
    // the cleanup mechanisms in our test suite will try to clean up webgl
    // textures, and if we mess with this flag then the physical texture types
    // will also change.

    const a = tf.tensor1d([1000, 2000, -2000, -4000]);
    const b = 70000;
    const result = tf.div(a, b);

    expect(result.shape).toEqual(a.shape);
    const resultData = await result.data();
    expectArraysClose(await resultData, [0.01429, 0.02857, -0.02857, -0.05714]);
  });
});
