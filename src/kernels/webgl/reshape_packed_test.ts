/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {expectArraysClose, PACKED_ENVS} from '../../test_util';

describeWithFlags('expensive reshape', PACKED_ENVS, () => {
  const cValues =
      [46, 52, 58, 64, 70, 100, 115, 130, 145, 160, 154, 178, 202, 226, 250];
  let c: tf.Tensor;

  beforeEach(() => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
    const b = tf.tensor2d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [3, 5]);
    c = tf.matMul(a, b);
  });

  it('6d --> 1d', () => {
    const cAs6D = tf.reshape(c, [1, 1, 1, 3, 1, 5]);
    const cAs1D = tf.reshape(cAs6D, [-1, cValues.length]);
    expectArraysClose(cAs1D, cValues);
  });
  it('1d --> 2d', () => {
    const cAs1D = tf.reshape(c, [cValues.length]);
    const cAs2D = tf.reshape(cAs1D, [5, -1]);
    expectArraysClose(cAs2D, cValues);
  });
  it('2d --> 3d', () => {
    const cAs3D = tf.reshape(c, [3, 1, 5]);
    expectArraysClose(cAs3D, cValues);
  });
  it('3d --> 4d', () => {
    const cAs3D = tf.reshape(c, [3, 1, 5]);
    const cAs4D = tf.reshape(cAs3D, [3, 5, 1, 1]);
    expectArraysClose(cAs4D, cValues);
  });
  it('4d --> 5d', () => {
    const cAs4D = tf.reshape(c, [3, 5, 1, 1]);
    const cAs5D = tf.reshape(cAs4D, [1, 1, 1, 5, 3]);
    expectArraysClose(cAs5D, cValues);
  });
  it('5d --> 6d', () => {
    const cAs5D = tf.reshape(c, [1, 1, 1, 5, 3]);
    const cAs6D = tf.reshape(cAs5D, [3, 5, 1, 1, 1, 1]);
    expectArraysClose(cAs6D, cValues);
  });
});

describeWithFlags('expensive reshape with even columns', PACKED_ENVS, () => {
  it('2 --> 4 columns', () => {
    const maxTextureSize = tf.ENV.get('WEBGL_MAX_TEXTURE_SIZE');

    let values: number[] = new Array<number>(16).fill(0);
    values = values.map((d, i) => i + 1);
    const a = tf.tensor2d(values, [8, 2]);
    const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', 2);
    // Setting WEBGL_MAX_TEXTURE_SIZE to 2 makes that [8, 2] tensor is packed
    // to texture of width 2 by height 2. Indices are packed as:
    // -------------
    // | 0 1 | 4 5 |       // First row's four
    // | 2 3 | 6 7 |       // pixels.
    // -------------
    // ...
    const c = tf.matMul(a, b);
    let cAs4D = c.reshape([2, 1, 2, 4]);
    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', maxTextureSize);

    // Execute non-packed operations to unpack tensor.
    const webglPackFlagSaved = tf.ENV.get('WEBGL_PACK');
    tf.ENV.set('WEBGL_PACK', false);
    cAs4D = cAs4D.add(1);
    cAs4D = cAs4D.add(-1);
    tf.ENV.set('WEBGL_PACK', webglPackFlagSaved);

    const result =
        [7, 10, 15, 22, 23, 34, 31, 46, 39, 58, 47, 70, 55, 82, 63, 94];
    expectArraysClose(cAs4D, result);
  });
});
