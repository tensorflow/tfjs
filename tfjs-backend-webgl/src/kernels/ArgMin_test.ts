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
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {WEBGL_ENVS} from '../backend_webgl_test_registry';

describeWithFlags('ArgMin', WEBGL_ENVS, () => {
  it('handles packed inputs', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);

    // pack a using the add op which packs outputs
    tf.env().set('WEBGL_PACK', true);
    const aPacked = tf.addN([a, tf.zeros(a.shape)]);

    tf.test_util.expectArraysEqual(await tf.argMin(aPacked).data(), [0, 1, 0]);
  });
});
