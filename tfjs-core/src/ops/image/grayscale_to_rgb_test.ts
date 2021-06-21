/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';

describeWithFlags('grayscaleToRGB', ALL_ENVS, () => {
  it('should convert (,1,3,1) images into (,1,3,3)', async () => {
    const grayscale = [[[[1.0], [2.0], [3.0]]]];

    const rgb = tf.image.grayscaleToRGB(grayscale);
    const rgbData = await rgb.data();

    const expected = [[[[1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                        [3.0, 3.0, 3.0]]]];

    expectArraysClose(expected, rgbData);
  });
});
