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
import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysEqual} from '../test_util';

describeWithFlags('nonMaxSuppressionPadded', ALL_ENVS, () => {
  it('select from three clusters with pad five.', async () => {
    const boxes = tf.tensor2d(
        [
          0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
          0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
        ],
        [6, 4]);
    const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
    const maxOutputSize = 5;
    const iouThreshold = 0.5;
    const scoreThreshold = 0.0;

    const {selectedIndices, validOutputs} = tf.image.nonMaxSuppressionPadded(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, true);

    expectArraysEqual(await selectedIndices.data(), [3, 0, 5, 0, 0]);
    expectArraysEqual(await validOutputs.data(), 3);
  });

  it('select from three clusters with pad five and score threshold.',
     async () => {
       const boxes = tf.tensor2d(
           [
             0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
             0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
           ],
           [6, 4]);
       const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
       const maxOutputSize = 6;
       const iouThreshold = 0.5;
       const scoreThreshold = 0.4;

       const before = tf.memory().numTensors;
       const {selectedIndices, validOutputs} = tf.image.nonMaxSuppressionPadded(
           boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, true);
       const after = tf.memory().numTensors;

       expectArraysEqual(await selectedIndices.data(), [3, 0, 0, 0, 0, 0]);
       expectArraysEqual(await validOutputs.data(), 2);
       expect(after).toEqual(before + 2);
     });

  it('select from three clusters with no padding when pad option is false.',
     async () => {
       const boxes = tf.tensor2d(
           [
             0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
             0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
           ],
           [6, 4]);
       const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
       const maxOutputSize = 5;
       const iouThreshold = 0.5;
       const scoreThreshold = 0.0;

       const {selectedIndices, validOutputs} = tf.image.nonMaxSuppressionPadded(
           boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, false);

       expectArraysEqual(await selectedIndices.data(), [3, 0, 5]);
       expectArraysEqual(await validOutputs.data(), 3);
     });
});
