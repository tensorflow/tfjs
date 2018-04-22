/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, expectArraysClose} from '../test_util';
import {describeWithFlags} from '../jasmine_util';

describeWithFlags('resizeNearestNeighbor', ALL_ENVS, () => {
  it('simple alignCorners=false', () => {
    const input = tf.tensor3d([2, 2, 4, 4], [2, 2, 1]);
    const output = input.resizeNearestNeighbor([3, 3], false);

    expectArraysClose(output, [2, 2, 2, 4, 4, 4, 4, 4, 4]);
  });

  it('simple alignCorners=true', () => {
    const input = tf.tensor3d([2, 2, 4, 4], [2, 2, 1]);
    const output = input.resizeNearestNeighbor([3, 3], true);

    expectArraysClose(output, [2, 2, 2, 4, 4, 4, 4, 4, 4]);
  });

  it('matches tensorflow w/ random numbers alignCorners=false', () => {
    const input = tf.tensor3d(
        [
          1.19074044, 0.91373104, 2.01611669, -0.52270832, 0.38725395,
          1.30809779, 0.61835143, 3.49600659, 2.09230986, 0.56473997,
          0.03823943, 1.19864896
        ],
        [2, 3, 2]);
    const output = input.resizeNearestNeighbor([4, 5], false);

    expectArraysClose(output, [
      1.19074044, 0.91373104, 2.01611669, -0.52270832, 2.01611669, -0.52270832,
      0.38725395, 1.30809779, 0.38725395, 1.30809779,  0.61835143, 3.49600659,
      2.09230986, 0.56473997, 2.09230986, 0.56473997,  0.03823943, 1.19864896,
      0.03823943, 1.19864896, 0.61835143, 3.49600659,  2.09230986, 0.56473997,
      2.09230986, 0.56473997, 0.03823943, 1.19864896,  0.03823943, 1.19864896,
      0.61835143, 3.49600659, 2.09230986, 0.56473997,  2.09230986, 0.56473997,
      0.03823943, 1.19864896, 0.03823943, 1.19864896
    ]);
  });

  it('matches tensorflow w/ random numbers alignCorners=true', () => {
    const input = tf.tensor3d(
        [
          1.19074044, 0.91373104, 2.01611669, -0.52270832, 0.38725395,
          1.30809779, 0.61835143, 3.49600659, 2.09230986, 0.56473997,
          0.03823943, 1.19864896
        ],
        [2, 3, 2]);
    const output = input.resizeNearestNeighbor([4, 5], true);

    expectArraysClose(output, [
      1.19074044, 0.91373104,  2.01611669, -0.52270832, 2.01611669, -0.52270832,
      0.38725395, 1.30809779,  0.38725395, 1.30809779,  1.19074044, 0.91373104,
      2.01611669, -0.52270832, 2.01611669, -0.52270832, 0.38725395, 1.30809779,
      0.38725395, 1.30809779,  0.61835143, 3.49600659,  2.09230986, 0.56473997,
      2.09230986, 0.56473997,  0.03823943, 1.19864896,  0.03823943, 1.19864896,
      0.61835143, 3.49600659,  2.09230986, 0.56473997,  2.09230986, 0.56473997,
      0.03823943, 1.19864896,  0.03823943, 1.19864896
    ]);
  });

  it('batch of 2, simple, alignCorners=true', () => {
    const input = tf.tensor4d([2, 2, 4, 4, 3, 3, 5, 5], [2, 2, 2, 1]);
    const output = input.resizeNearestNeighbor([3, 3], true /* alignCorners */);

    expectArraysClose(
        output, [2, 2, 2, 4, 4, 4, 4, 4, 4, 3, 3, 3, 5, 5, 5, 5, 5, 5]);
  });

  it('throws when passed a non-tensor', () => {
    const e =
        /Argument 'images' passed to 'resizeNearestNeighbor' must be a Tensor/;
    expect(() => tf.image.resizeNearestNeighbor({} as tf.Tensor3D, [
      1, 1
    ])).toThrowError(e);
  });
});
