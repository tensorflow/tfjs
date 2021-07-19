/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {memory, test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as tf from './index';
import {RN_ENVS} from './test_env_registry';

const uint8array = new Uint8Array([
  255, 216, 255, 224, 0,   16,  74,  70,  73,  70,  0,   1,   1,   2,   0,
  118, 0,   118, 0,   0,   255, 219, 0,   67,  0,   1,   1,   1,   1,   1,
  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   255,
  219, 0,   67,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
  1,   1,   1,   1,   1,   1,   1,   1,   255, 194, 0,   17,  8,   0,   2,
  0,   2,   3,   1,   17,  0,   2,   17,  1,   3,   17,  1,   255, 196, 0,
  20,  0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  0,   0,   0,   7,   255, 196, 0,   20,  1,   1,   0,   0,   0,   0,   0,
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   8,   255, 218, 0,   12,
  3,   1,   0,   2,   16,  3,   16,  0,   0,   1,   50,  53,  173, 255, 0,
  255, 196, 0,   21,  16,  1,   1,   0,   0,   0,   0,   0,   0,   0,   0,
  0,   0,   0,   0,   0,   0,   5,   4,   255, 218, 0,   8,   1,   1,   0,
  1,   5,   2,   60,  195, 104, 131, 255, 196, 0,   25,  17,  0,   3,   1,
  1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,   3,
  4,   5,   1,   0,   255, 218, 0,   8,   1,   3,   1,   1,   63,  1,   159,
  15,  18,  201, 209, 93,  120, 249, 117, 87,  82,  85,  77,  85,  83,  159,
  35,  232, 166, 135, 128, 181, 207, 123, 154, 162, 99,  156, 230, 17,  49,
  173, 97,  17,  176, 200, 136, 139, 165, 222, 247, 223, 255, 196, 0,   26,
  17,  0,   3,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,
  0,   0,   1,   2,   3,   4,   5,   35,  0,   255, 218, 0,   8,   1,   2,
  1,   1,   63,  1,   29,  110, 166, 80,  51,  102, 233, 116, 51,  230, 207,
  225, 159, 60,  54,  104, 148, 33,  9,   121, 202, 49,  148, 232, 169, 41,
  73,  21,  82,  115, 69,  8,   136, 2,   168, 0,   1,   247, 255, 196, 0,
  25,  16,  0,   3,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,
  0,   0,   0,   1,   2,   3,   4,   0,   5,   255, 218, 0,   8,   1,   1,
  0,   6,   63,  2,   197, 163, 71,  159, 134, 247, 190, 76,  214, 189, 237,
  146, 21,  181, 173, 88,  163, 210, 181, 163, 161, 122, 82,  142, 75,  187,
  185, 44,  204, 75,  49,  36,  247, 255, 196, 0,   21,  16,  1,   1,   0,
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,
  255, 218, 0,   8,   1,   1,   0,   1,   63,  33,  85,  47,  2,   3,   60,
  155, 126, 1,   191, 255, 218, 0,   12,  3,   1,   0,   2,   0,   3,   0,
  0,   0,   16,  223, 255, 196, 0,   20,  17,  1,   0,   0,   0,   0,   0,
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   255, 218, 0,   8,
  1,   3,   1,   1,   63,  16,  117, 250, 0,   169, 238, 184, 84,  212, 255,
  196, 0,   21,  17,  1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  0,   0,   0,   0,   0,   1,   17,  255, 218, 0,   8,   1,   2,   1,   1,
  63,  16,  8,   234, 1,   198, 37,  68,  148, 101, 31,  255, 196, 0,   20,
  16,  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  0,   0,   1,   255, 218, 0,   8,   1,   1,   0,   1,   63,  16,  111, 197,
  200, 77,  239, 120, 73,  43,  255, 217
]);

describeWithFlags('decode images', RN_ENVS, () => {
  it('decode jpg', async () => {
    const beforeNumTensors: number = memory().numTensors;
    const imageTensor = await tf.decodeJpeg(uint8array);
    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([2, 2, 3]);
    test_util.expectArraysEqual(
        await imageTensor.data(),
        [240, 100, 0, 50, 50, 50, 99, 49, 0, 199, 99, 49]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });
});
