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

import * as tf from '../../index';
import {describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';
import {PACKED_ENVS, WEBGL_ENVS} from './backend_webgl_test_registry';

describeWithFlags('batchNorm', WEBGL_ENVS, () => {
  it('should work for broadcasted inputs', async () => {
    const x = tf.tensor4d([2, 4, 9, 23], [2, 1, 1, 2]);
    const mean = tf.tensor4d([1], [1, 1, 1, 1]);
    const variance = tf.tensor4d([1], [1, 1, 1, 1]);

    const result = tf.batchNorm4d(x, mean, variance);
    expectArraysClose(
        await result.data(), [0.9995003, 2.9985011, 7.9960027, 21.9890079]);
  });

  it('should work when squarification results in zero padding', async () => {
    const maxTextureSize = tf.ENV.getNumber('WEBGL_MAX_TEXTURE_SIZE');
    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', 5);

    const x = tf.tensor3d(
        [
          0.49955603, 0.04158615, -1.09440524, 2.03854165, -0.61578344,
          2.87533573, 1.18105987, 0.807462, 1.87888837, 2.26563962, -0.37040935,
          1.35848753, -0.75347094, 0.15683117, 0.91925946, 0.34121279,
          0.92717143, 1.89683965
        ],
        [2, 3, 3]);
    const mean = tf.tensor1d([0.39745062, -0.48062894, 0.4847822]);
    const variance = tf.tensor1d([0.32375343, 0.67117643, 1.08334653]);
    const offset = tf.tensor1d([0.69398749, -1.29056387, 0.9429723]);
    const scale = tf.tensor1d([-0.5607271, 0.9878457, 0.25181573]);
    const varianceEpsilon = .001;

    const result =
        tf.batchNorm3d(x, mean, variance, offset, scale, varianceEpsilon);

    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', maxTextureSize);

    expectArraysClose(await result.data(), [
      0.59352049, -0.66135202, 0.5610874, -0.92077015, -1.45341019, 1.52106473,
      -0.07704776, 0.26144429, 1.28010017, -1.14422404, -1.15776136, 1.15425493,
      1.82644104, -0.52249442, 1.04803919, 0.74932291, 0.40568101, 1.2844412
    ]);
  });
});

describeWithFlags('batchnorm packed', PACKED_ENVS, () => {
  it('should not leak memory', () => {
    const x = tf.tensor4d([2, 4, 9, 23], [2, 1, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const varianceEpsilon = .001;

    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    tf.batchNorm4d(x, mean, variance, undefined, undefined, varianceEpsilon);
    const endNumBytes = tf.memory().numBytes;
    const endNumTensors = tf.memory().numTensors;

    expect(endNumBytes - startNumBytes).toEqual(16);
    expect(endNumTensors - startNumTensors).toEqual(1);
  });
});
