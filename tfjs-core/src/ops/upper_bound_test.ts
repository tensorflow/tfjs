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

describeWithFlags('upperBound', ALL_ENVS, () => {
  it('test1D', async () => {
    // Tests against numpy generated data.
    const NUMPY_DATA = {
      'float32': [
        [
          -725.0505981445312,  -721.4473266601562, -669.2916259765625,
          -460.14422607421875, -304.4682922363281, -302.20330810546875,
          -204.64633178710938, -143.817626953125,  243.3914337158203,
          247.34442138671875,  326.88299560546875, 451.9959716796875,
          501.62420654296875,  501.8848571777344,  614.7825927734375,
          766.6121826171875,   791.7724609375,     806.8038330078125,
          855.0171508789062,   929.6801147460938
        ],
        [
          -795.3311157226562, -171.88803100585938, 388.8003234863281,
          -171.64146423339844, -900.0930786132812, 71.79280853271484,
          327.58929443359375, 29.77822494506836, 889.1895141601562,
          173.11007690429688
        ],
        [0, 7, 11, 7, 0, 8, 11, 8, 19, 8]
      ],
      'int32': [
        [
          -968, -867, -751, -725, -655, -346, -285, 54,  246, 381,
          393,  423,  507,  510,  771,  817,  846,  858, 865, 994
        ],
        [-770, 898, -100, 156, -183, -525, 806, 147, -994, 234],
        [2, 19, 7, 8, 7, 5, 15, 8, 0, 8]
      ]
    };
    for (const dtype of ['float32', 'int32'] as const ) {
      const [sortedSequence, values, npAns] = NUMPY_DATA[dtype];

      const result = tf.upperBound(sortedSequence, values);

      expectArraysClose(await result.data(), npAns);
    }
  });

  it('upperBound2D', async () => {
    for (const dtype of ['float32', 'int32'] as const ) {
      const sortedSequence =
          tf.tensor2d([[0, 3, 9, 9, 10], [1, 2, 3, 4, 5]], undefined, dtype);
      const values = tf.tensor2d([[2, 4, 9], [0, 2, 6]], undefined, dtype);
      const correctAns = [[1, 2, 4], [0, 2, 5]];

      const result = tf.upperBound(sortedSequence, values);

      expectArraysClose(await result.data(), correctAns);
    }
  });
});
