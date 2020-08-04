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
import {describeWithFlags, ALL_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

describeWithFlags('SpaceToBatchND.', ALL_ENVS, () => {
  it('has no memory leak.', async () => {
    const initialDataIds = tf.engine().backend.numDataIds();

    const t = tf.tensor4d([[[[1], [2]], [[3], [4]]]], [1, 2, 2, 1]);
    const blockShape = [2, 2];
    const paddings = [[0, 0], [0, 0]];

    const res = tf.spaceToBatchND(t, blockShape, paddings);
    expect(res.shape).toEqual([4, 1, 1, 1]);
    expectArraysClose(await res.data(), [1, 2, 3, 4]);

    const afterResDataIds = tf.engine().backend.numDataIds();
    // 1 input tensor and 1 result tensor
    expect(afterResDataIds).toEqual(initialDataIds + 2);

    t.dispose();
    res.dispose();

    const afterDisposeDataIds = tf.engine().backend.numDataIds();

    expect(afterDisposeDataIds).toEqual(initialDataIds);
  });
});
