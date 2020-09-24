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
import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysClose, expectArraysEqual} from '../../test_util';

describeWithFlags('nonMaxSuppressionAsync', ALL_ENVS, () => {
  describe('NonMaxSuppressionAsync basic', () => {
    it('select from three clusters', async () => {
      const boxes = tf.tensor2d(
          [
            0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
            0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
          ],
          [6, 4]);
      const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
      const maxOutputSize = 3;
      const iouThreshold = 0.5;
      const scoreThreshold = 0;
      const indices = await tf.image.nonMaxSuppressionAsync(
          boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

      expect(indices.shape).toEqual([3]);
      expectArraysEqual(await indices.data(), [3, 0, 5]);
    });

    it('accepts a tensor-like object', async () => {
      const boxes = [[0, 0, 1, 1], [0, 1, 1, 2]];
      const scores = [1, 2];
      const indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, 10);
      expect(indices.shape).toEqual([2]);
      expect(indices.dtype).toEqual('int32');
      expectArraysEqual(await indices.data(), [1, 0]);
    });
  });

  describe('NonMaxSuppressionWithScoreAsync', () => {
    it('select from three clusters with SoftNMS', async () => {
      const boxes = tf.tensor2d(
          [
            0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
            0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
          ],
          [6, 4]);
      const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
      const maxOutputSize = 6;
      const iouThreshold = 1.0;
      const scoreThreshold = 0;
      const softNmsSigma = 0.5;

      const numTensorsBefore = tf.memory().numTensors;

      const {selectedIndices, selectedScores} =
          await tf.image.nonMaxSuppressionWithScoreAsync(
              boxes, scores, maxOutputSize, iouThreshold, scoreThreshold,
              softNmsSigma);

      const numTensorsAfter = tf.memory().numTensors;

      expectArraysEqual(await selectedIndices.data(), [3, 0, 1, 5, 4, 2]);

      expectArraysClose(
          await selectedScores.data(), [0.95, 0.9, 0.384, 0.3, 0.256, 0.197]);

      // The number of tensors should increase by the number of tensors
      // returned (i.e. selectedIndices and selectedScores).
      expect(numTensorsAfter).toEqual(numTensorsBefore + 2);
    });
  });

  describe('NonMaxSuppressionPaddedAsync', () => {
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

      const before = tf.memory().numTensors;

      const {selectedIndices, validOutputs} =
          await tf.image.nonMaxSuppressionPaddedAsync(
              boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, true);

      const after = tf.memory().numTensors;

      expectArraysEqual(await selectedIndices.data(), [3, 0, 5, 0, 0]);
      expectArraysEqual(await validOutputs.data(), 3);

      // The number of tensors should increase by the number of tensors
      // returned (i.e. selectedIndices and selectedScores).
      expect(after).toEqual(before + 2);
    });
  });
});
