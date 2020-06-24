/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {Tensor} from '../tensor';
import {expectArraysClose} from '../test_util';

describeWithFlags('broadcastTo', ALL_ENVS, () => {
  it('[] -> [3,2]', async () => {
    const a = tf.scalar(4.2);
    const A = tf.tensor2d([[4.2, 4.2], [4.2, 4.2], [4.2, 4.2]]);

    expectArraysClose(
        await A.array(), await tf.broadcastTo(a, A.shape).array());

    // test gradients
    const w = tf.tensor2d([[4.7, 4.5], [-6.1, -6.6], [-8.1, -3.4]]),
          f = (a: Tensor) =>
              tf.broadcastTo(a, A.shape).mul(w).mean().asScalar(),
          h = (a: Tensor) => a.mul(w).mean().asScalar();

    const df = tf.grad(f), dh = tf.grad(h);

    expectArraysClose(await df(a).array(), await dh(a).array());
  });

  it('[2] -> [3,2]', async () => {
    const a = tf.tensor1d([1, 2]);
    const A = tf.tensor2d([[1, 2], [1, 2], [1, 2]]);
    expectArraysClose(
        await A.array(), await tf.broadcastTo(a, A.shape).array());

    // test gradients
    const w = tf.tensor2d([[4.7, 4.5], [-6.1, -6.6], [-8.1, -3.4]]),
          f = (a: Tensor) =>
              tf.broadcastTo(a, A.shape).mul(w).mean().asScalar(),
          h = (a: Tensor) => a.mul(w).mean().asScalar();

    const df = tf.grad(f), dh = tf.grad(h);

    expectArraysClose(await df(a).array(), await dh(a).array());
  });

  it('[3,1] -> [3,2]', async () => {
    const a = tf.tensor2d([[1], [2], [3]]);
    const A = tf.tensor2d([[1, 1], [2, 2], [3, 3]]);

    expectArraysClose(
        await A.array(), await tf.broadcastTo(a, A.shape).array());

    // test gradients
    const w = tf.tensor2d([[4.7, 4.5], [-6.1, -6.6], [-8.1, -3.4]]),
          f = (a: Tensor) =>
              tf.broadcastTo(a, A.shape).mul(w).mean().asScalar(),
          h = (a: Tensor) => a.mul(w).mean().asScalar();

    const df = tf.grad(f), dh = tf.grad(h);

    expectArraysClose(await df(a).array(), await dh(a).array());
  });
});
