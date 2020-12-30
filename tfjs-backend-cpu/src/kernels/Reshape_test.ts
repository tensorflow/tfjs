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
import {Tensor, test_util} from '@tensorflow/tfjs-core';

const {expectArraysClose, expectArraysEqual} = test_util;
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags, ALL_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

describeWithFlags('Reshape.', ALL_ENVS, () => {
  it('does not have memory leak.', async () => {
    const beforeDataIds = tf.engine().backend.numDataIds();

    const x = tf.tensor1d([1, 1, 1, 1]);
    const res =
        // tslint:disable-next-line: no-unnecessary-type-assertion
        tf.engine().runKernel('Reshape', {x}, {shape: [2, 2]}) as Tensor;

    expectArraysClose(await res.data(), [1, 1, 1, 1]);
    expectArraysEqual(res.shape, [2, 2]);

    const afterResDataIds = tf.engine().backend.numDataIds();
    expect(afterResDataIds).toEqual(beforeDataIds + 1);

    x.dispose();
    res.dispose();

    const afterDisposeDataIds = tf.engine().backend.numDataIds();
    expect(afterDisposeDataIds).toEqual(beforeDataIds);
  });

  it('does not have memory leak calling reshape twice.', async () => {
    const beforeDataIds = tf.engine().backend.numDataIds();

    // Adding 1 new dataId.
    const x = tf.tensor1d([1, 1, 1, 1]);

    // Does not add new dataId;
    const res =
        // tslint:disable-next-line: no-unnecessary-type-assertion
        tf.engine().runKernel('Reshape', {x}, {shape: [2, 2]}) as Tensor;

    expectArraysEqual(res.shape, [2, 2]);

    // Does not add new dataId.
    const res2 =
        // tslint:disable-next-line: no-unnecessary-type-assertion
        tf.engine().runKernel('Reshape', {x: res}, {shape: [1, 4]}) as Tensor;
    expectArraysEqual(res2.shape, [1, 4]);

    const afterRes2DataIds = tf.engine().backend.numDataIds();
    expect(afterRes2DataIds).toEqual(beforeDataIds + 1);

    res.dispose();

    const afterResDataIds = tf.engine().backend.numDataIds();
    expect(afterResDataIds).toEqual(beforeDataIds + 1);

    x.dispose();
    res2.dispose();

    const afterDisposeDataIds = tf.engine().backend.numDataIds();
    // Should be able to dispose the dataId.
    expect(afterDisposeDataIds).toEqual(beforeDataIds);
  });
});
