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


    it('throws Error if blockSize < 2', () => {
      const x = tf.tensor4d([[[[1], [2]], [[3], [4]]]]);
      const blockSize = 1;
      const dataFormat = 'NHWC';
      expect(() => tf.spaceToDepth(x, blockSize, dataFormat))
          .toThrowError(`Block size must be >= 2, but is ${blockSize}`);
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
              `Input height must be evenly divisible by block size  , but is ${
                  x.shape[1]} for inputHeight and ${blockSize} for blockSize`);
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
              `Input width must be evenly divisible by block size  , but is ${
                  x.shape[2]} for inputWidth and ${blockSize} for blockSize`);
    })
  });
});
