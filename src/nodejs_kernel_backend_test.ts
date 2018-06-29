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

import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line:max-line-length
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

describe('delayed upload', () => {
  it('should handle data before op execution', () => {
    const t = tf.tensor1d([1, 2, 3]);
    expectArraysClose(t, [1, 2, 3]);

    const r = t.add(tf.tensor1d([4, 5, 6]));
    expectArraysClose(r, [5, 7, 9]);
  });

  it('Should not cache tensors in the tensor map for device support. ', () => {
    const logits = tf.tensor1d([1, 2, 3]);
    const softmaxLogits = tf.softmax(logits);
    const data = softmaxLogits.dataSync();
    expect(softmaxLogits.get(0)).toEqual(data[0]);
    expect(softmaxLogits.get(1)).toEqual(data[1]);
    expect(softmaxLogits.get(2)).toEqual(data[2]);
  });
});

describe('type casting', () => {
  it('exp support int32', () => {
    tf.exp(tf.scalar(2, 'int32'));
  });
});
