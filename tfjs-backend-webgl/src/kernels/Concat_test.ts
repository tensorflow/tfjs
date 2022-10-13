/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import {expectArraysClose} from '@tensorflow/tfjs-core/test_util';

import {WEBGL_ENVS} from '../backend_webgl_test_registry';

describeWithFlags('Concat', WEBGL_ENVS, () => {
  it('Works if input size is larger than WEBGL_MAX_TEXTURES_IN_SHADER',
     async () => {
       const x1 = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
       const tensorNum = tf.env().getNumber('WEBGL_MAX_TEXTURES_IN_SHADER') + 1;
       const input = [];
       const expected = [];
       for (let i = 0; i < tensorNum; i++) {
         input.push(x1);
         expected.push(1, 2, 3, 4);
       }
       const values = tf.concat(input, 0);
       expect(values.shape).toEqual([tensorNum * 2, 2, 1]);
       expectArraysClose(await values.data(), expected);
     });
});
