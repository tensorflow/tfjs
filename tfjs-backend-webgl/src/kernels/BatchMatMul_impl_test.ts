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

import * as tf from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {MathBackendWebGL} from '../backend_webgl';
import {WEBGL_ENVS} from '../backend_webgl_test_registry';

import {batchMatMulImpl} from './BatchMatMul_impl';

const {expectArraysClose} = test_util;

describeWithFlags('batchMatMulImpl', WEBGL_ENVS, () => {
  it('(A x B).T', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 4, 3]);
    const b = tf.tensor3d([0, 1, -3, 2, 2, 1], [1, 3, 2]);
    const transposeA = false;
    const transposeB = false;
    const transposeProduct = true;

    const result = batchMatMulImpl({
      a,
      b,
      transposeA,
      transposeB,
      backend: tf.backend() as MathBackendWebGL,
      transposeProduct,
    });

    expect(result.shape).toEqual([1, 2, 4]);
    expectArraysClose(
        tf.backend().readSync(result.dataId), [0, -3, -6, -9, 8, 20, 32, 44]);
  });
});
