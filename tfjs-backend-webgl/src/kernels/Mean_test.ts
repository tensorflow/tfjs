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

describeWithFlags('Mean.', ALL_ENVS, () => {
  it('Multi-stage mean produces the correct result and does not leak memory.',
     async () => {
       const MAX_FLOAT16 = 300;
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

       const aData = new Float32Array(MAX_FLOAT16 + 1);
       for (let i = 0; i < aData.length; i++) {
         aData[i] = i;
       }
       const a = tf.tensor1d(aData);
       const r = tf.mean(a);

       expect(r.dtype).toBe('float32');
       expectArraysClose(await r.data(), 150);
     });
});
