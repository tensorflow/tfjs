/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

describeWithFlags('lowerBound', ALL_ENVS, () => {
  it('test1D', async () => {
    // Tests against numpy generated data.
    const NUMPY_DATA = {
      'float32': [
        [
          -945.2247924804688,  -921.8904418945312, -829.9115600585938,
          -719.2261352539062,  -660.3391723632812, -603.7969970703125,
          -591.0955200195312,  -373.1516418457031, -165.39039611816406,
          -161.61097717285156, 117.37965393066406, 340.9350280761719,
          370.4389953613281,   384.6452331542969,  601.4891357421875,
          752.7783203125,      756.23486328125,    756.2850341796875,
          789.2133178710938,   936.5231323242188
        ],
        [
          -165.95599365234375, 440.64898681640625, -999.771240234375,
          -395.3348693847656, -706.4882202148438, -815.3228149414062,
          -627.4795532226562, -308.8785400390625, -206.46505737304688,
          77.63346862792969
        ],
        [8, 14, 0, 7, 4, 3, 5, 8, 8, 10]
      ],
      'int32': [
        [
          -961, -893, -793, -739, -706, -576, -468, -439, -424, -412,
          -104, -16,  148,  178,  357,  399,  496,  578,  817,  977
        ],
        [-803, -157, 915, 66, 383, -368, 373, 669, -963, 500],
        [2, 10, 19, 12, 15, 10, 15, 18, 0, 17]
      ],
    };
    for (const dtype of ['float32', 'int32'] as const ) {
      const [sortedSequence, values, npAns] = NUMPY_DATA[dtype];

      const result = tf.lowerBound(sortedSequence, values);

      expectArraysClose(await result.data(), npAns);
    }
  });

  it('lowerBound2D', async () => {
    for (const dtype of ['float32', 'int32'] as const ) {
      const sortedSequence =
          tf.tensor2d([[0, 3, 9, 9, 10], [1, 2, 3, 4, 5]], undefined, dtype);
      const values = tf.tensor2d([[2, 4, 9], [0, 2, 6]], undefined, dtype);
      const correctAns = [[1, 2, 2], [0, 1, 5]];

      const result = tf.lowerBound(sortedSequence, values);

      expectArraysClose(await result.data(), correctAns);
    }
  });
});
