/**
 * @license
 * Copyright 2022 Google LLC.
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
import {expectArraysClose} from '../test_util';

/**
 * Test utility for testing AvgPool, MaxPool, etc where kernel size is 1x1,
 * effectively making them act as the identity function except where strides
 * affect the output.
 */
export function identityPoolTest(pool: typeof tf.avgPool) {
  it('1x1 pool size (identity)', async () => {
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const a = tf.range(0, 10).reshape([1, 1, 1, 10]) as tf.Tensor4D;
    const result = pool(a, [1, 1], [1, 1], 'valid');
    expectArraysClose(await result.data(), await a.data());
  });

  it('1x1 pool size with strides', async () => {
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const a = tf.range(0, 150).reshape([1, 10, 15, 1]) as tf.Tensor4D;
    const result = pool(a, [1, 1], [3, 4], 'valid');
    expectArraysClose(await result.data(), [
      0, 4, 8, 12,
      45, 49, 53, 57,
      90, 94, 98, 102,
      135, 139, 143, 147,
    ]);
  });

  it('1x1 pool size batched', async () => {
    // 7 batches of 3 x 4
    const shape = [7, 3, 4, 1];
    const size = shape.reduce((a, b) => a * b, 1);
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const a = tf.range(0, size).reshape(shape) as tf.Tensor4D;
    const result = pool(a, [1, 1], [1, 1], 'valid');
    expectArraysClose(await result.data(), await a.data());
  });

  it('1x1 pool size batched with strides', async () => {
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const a = tf.range(0, 300).reshape([2, 10, 15, 1]) as tf.Tensor4D;
    const result = pool(a, [1, 1], [3, 4], 'valid');
    expectArraysClose(await result.data(), [
      // Batch 0
      0, 4, 8, 12,
      45, 49, 53, 57,
      90, 94, 98, 102,
      135, 139, 143, 147,
      // Batch 1
      150, 154, 158, 162,
      195, 199, 203, 207,
      240, 244, 248, 252,
      285, 289, 293, 297,
    ]);
  });

  it('1x1 pool size batched with strides and channels', async () => {
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const a = tf.range(0, 900).reshape([2, 10, 15, 3]) as tf.Tensor4D;
    const result = pool(a, [1, 1], [3, 4], 'valid');
    expectArraysClose(await result.data(), [
      // Batch 0
      0, 1, 2, 12, 13, 14, 24, 25, 26, 36, 37, 38,
      135, 136, 137, 147, 148, 149, 159, 160, 161, 171, 172, 173,
      270, 271, 272, 282, 283, 284, 294, 295, 296, 306, 307, 308,
      405, 406, 407, 417, 418, 419, 429, 430, 431, 441, 442, 443,
      // Batch 1
      450, 451, 452, 462, 463, 464, 474, 475, 476, 486, 487, 488,
      585, 586, 587, 597, 598, 599, 609, 610, 611, 621, 622, 623,
      720, 721, 722, 732, 733, 734, 744, 745, 746, 756, 757, 758,
      855, 856, 857, 867, 868, 869, 879, 880, 881, 891, 892, 893,
    ]);
  });
}
