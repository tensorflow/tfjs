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
import {expectArraysClose} from '../test_util';

describeWithFlags('spaceToDepth', ALL_ENVS, () => {
  describe('should produce the correct output', () => {
    it('for inputShape=[1,2,2,1], blockSize=2, dataFormat=NHWC', async () => {
      const x = tf.tensor4d([[[[1], [2]], [[3], [4]]]]);
      const blockSize = 2;
      const dataFormat = 'NHWC';
      const y = tf.spaceToDepth(x, blockSize, dataFormat);
      expect(y.shape).toEqual([1, 1, 1, 4])
      expectArraysClose(await y.data(), [[[[1, 2, 3, 4]]]])
    });

    it('for inputShape=[1,2,2,3], blockSize=2, dataFormat=NHWC', async () => {
      const x = tf.tensor4d([[
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
      ]]);
      const blockSize = 2;
      const dataFormat = 'NHWC';
      const y = tf.spaceToDepth(x, blockSize, dataFormat);
      expect(y.shape).toEqual([1, 1, 1, 12]);
      expectArraysClose(
          await y.data(), [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]);
    });

    it('for inputShape=[1,4,4,1], blockSize=2, dataFormat=NHWC', async () => {
      const x = tf.tensor4d([[
        [[1], [2], [5], [6]],
        [[3], [4], [7], [8]],
        [[9], [10], [13], [14]],
        [[11], [12], [15], [16]],
      ]]);
      const blockSize = 2;
      const dataFormat = 'NHWC';
      const y = tf.spaceToDepth(x, blockSize, dataFormat);
      expect(y.shape).toEqual([1, 2, 2, 4]);
      expectArraysClose(await y.data(), [
        [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]
      ]);
    });

    it('for inputShape=[1,4,4,4], blockSize=2, dataFormat=NHWC', async () => {
      const x =
          tf.tensor4d(Array(64).fill(0).map((_, idx) => idx), [1, 4, 4, 4]);
      const blockSize = 2;
      const dataFormat = 'NHWC';
      const y = tf.spaceToDepth(x, blockSize, dataFormat);
      expect(y.shape).toEqual([1, 2, 2, 16]);
      expectArraysClose(
          await y.data(), [[
            [
              [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23],
              [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]
            ],
            [
              [32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55],
              [40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63]
            ]
          ]]);
    });

    it('throws Error if blockSize < 2', () => {
      const x = tf.tensor4d([[[[1], [2]], [[3], [4]]]]);
      const blockSize = 1;
      const dataFormat = 'NHWC';
      expect(() => tf.spaceToDepth(x, blockSize, dataFormat))
          .toThrowError(`blockSize must be >= 2, got ${blockSize}`);
    });

    it('throws Error if inputHeight is not divisible by blockSize', () => {
      const x = tf.tensor4d([[
        [[1], [2], [5], [6]],
        [[3], [4], [7], [8]],
        [[9], [10], [13], [14]],
        [[11], [12], [15], [16]],
        [[17], [18], [19], [20]],
      ]]);
      const blockSize = 2;
      const dataFormat = 'NHWC';
      expect(() => tf.spaceToDepth(x, blockSize, dataFormat))
          .toThrowError(
              `inputHeight must be divisible by blockSize, got inputHeight=${
                  x.shape[1]}, blockSize=${blockSize}`);
    });

    it('throws Error if inputWidth is not divisible by blockSize', () => {
      const x = tf.tensor4d([[
        [[1], [2], [5], [6], [17]],
        [[3], [4], [7], [8], [18]],
        [[9], [10], [13], [14], [19]],
        [[11], [12], [15], [16], [20]],
      ]]);
      const blockSize = 2;
      const dataFormat = 'NHWC';
      expect(() => tf.spaceToDepth(x, blockSize, dataFormat))
          .toThrowError(
              `inputWidth must be divisible by blockSize, got inputWidth=${
                  x.shape[2]}, blockSize=${blockSize}`);
    })
  });
});
