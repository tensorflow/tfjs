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
import {ALL_ENVS, BROWSER_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

describeWithFlags('depthToSpace', ALL_ENVS, () => {
  it('tensor4d, input shape=[1, 1, 1, 4], blockSize=2, format=NHWC',
     async () => {
       const t = tf.tensor4d([[[[1, 2, 3, 4]]]]);
       const blockSize = 2;
       const dataFormat = 'NHWC';

       const res = tf.depthToSpace(t, blockSize, dataFormat);
       expect(res.shape).toEqual([1, 2, 2, 1]);
       expectArraysClose(await res.data(), [1, 2, 3, 4]);
     });

  it('tensor4d, input shape=[1, 1, 1, 12], blockSize=2, format=NHWC',
     async () => {
       const t = tf.tensor4d([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]);
       const blockSize = 2;
       const dataFormat = 'NHWC';

       const res = tf.depthToSpace(t, blockSize, dataFormat);
       expect(res.shape).toEqual([1, 2, 2, 3]);
       expectArraysClose(
           await res.data(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
     });

  it('tensor4d, input shape=[1, 2, 2, 4], blockSize=2, format=NHWC',
     async () => {
       const t = tf.tensor4d([
         [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]
       ]);
       const blockSize = 2;
       const dataFormat = 'NHWC';

       const res = tf.depthToSpace(t, blockSize, dataFormat);
       expect(res.shape).toEqual([1, 4, 4, 1]);
       expectArraysClose(
           await res.data(),
           [1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]);
     });

  it('throws when depth not divisible by blockSize * blockSize', () => {
    const t = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
    const blockSize = 3;

    expect(() => tf.depthToSpace(t, blockSize))
        .toThrowError(`Dimension size must be evenly divisible by ${
            blockSize * blockSize} but is ${
            t.shape[3]} for depthToSpace with input shape ${t.shape}`);
  });
});

describeWithFlags('depthToSpace', BROWSER_ENVS, () => {
  it('throws when blocksize < 2', () => {
    const t = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
    const blockSize = 1;

    expect(() => tf.depthToSpace(t, blockSize))
        .toThrowError(
            `blockSize should be > 1 for depthToSpace, but was: ${blockSize}`);
  });
});
