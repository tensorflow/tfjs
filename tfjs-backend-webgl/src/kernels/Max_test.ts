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
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

describeWithFlags('Max', ALL_ENVS, () => {
  it('does not have memory leak when calling reduce multiple times.',
     async () => {
       const beforeDataIds = tf.engine().backend.numDataIds();

       // Input must be large enough to trigger multi-stage reduction.
       const x = tf.ones([100, 100]);
       const xMax = x.max();

       const afterResDataIds = tf.engine().backend.numDataIds();
       expect(afterResDataIds).toEqual(beforeDataIds + 2);

       x.dispose();
       xMax.dispose();

       const afterDisposeDataIds = tf.engine().backend.numDataIds();
       expect(afterDisposeDataIds).toEqual(beforeDataIds);
     });
});
