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
  it('Multi-stage div produces the correct result and does not leak memory.',
     async () => {
       const MAX_FLOAT16 = 65504;
       // We can't simply flip the WEBGL_RENDER_FLOAT32_ENABLED flag to test
       // this functionality because as part of test cleanup we dispose webgl
       // textures. WEBGL_RENDER_FLOAT32_ENABLED determines the physical type
       // of the textures we register / dispose, so by altering
       // WEBGL_RENDER_FLOAT32_ENABLED within a test we may end up trying to
       // clean up textures that were never registered.
       spyOn(webgl_util, 'canBeRepresented').and.callFake((val: number) => {
         if (val > MAX_FLOAT16) {
           return false;
         }
         return true;
       });

       const a = tf.tensor1d([1000, 2000, -2000, -4000]);
       const b = 70000;

       const nBeforeDataIds = tf.engine().backend.numDataIds();
       const result = tf.div(a, b);
       const nAfterDataIds = tf.engine().backend.numDataIds();

       expect(nAfterDataIds).toBe(nBeforeDataIds + 1);
       expect(result.shape).toEqual(a.shape);
       expectArraysClose(
           await result.data(), [0.01429, 0.02857, -0.02857, -0.05714]);
     });

  it('Multi-stage div produces the correct result in case of underflow.',
     async () => {
       const MIN_FLOAT16 = 0.1;
       spyOn(webgl_util, 'canBeRepresented').and.callFake((val: number) => {
         if (val < MIN_FLOAT16) {
           return false;
         }
         return true;
       });

       const a = tf.tensor1d([1, 2, -2, -4]);
       const b = 0.01;

       const nBeforeDataIds = tf.engine().backend.numDataIds();
       const result = tf.div(a, b);
       const nAfterDataIds = tf.engine().backend.numDataIds();

       expect(nAfterDataIds).toBe(nBeforeDataIds + 1);
       expect(result.shape).toEqual(a.shape);
       expectArraysClose(await result.data(), [100, 200, -200, -400]);
     });
});
