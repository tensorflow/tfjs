/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import * as test_util from '../test_util';
import {MathTests} from '../test_util';

import {Array3D} from './ndarray';

// math.resizeBilinear3D
{
  const tests: MathTests = it => {
    it('simple alignCorners=false', math => {
      const input = Array3D.new([2, 2, 1], [2, 2, 4, 4]);
      const output = math.resizeBilinear3D(input, [3, 3], false);

      test_util.expectArraysClose(
          output, [2, 2, 2, 10 / 3, 10 / 3, 10 / 3, 4, 4, 4]);
    });

    it('simple alignCorners=true', math => {
      const input = Array3D.new([2, 2, 1], [2, 2, 4, 4]);
      const output = math.resizeBilinear3D(input, [3, 3], true);

      test_util.expectArraysClose(output, [2, 2, 2, 3, 3, 3, 4, 4, 4]);
    });

    it('matches tensorflow w/ random numbers alignCorners=false', math => {
      const input = Array3D.new([2, 3, 2], [
        1.19074044, 0.91373104, 2.01611669, -0.52270832, 0.38725395, 1.30809779,
        0.61835143, 3.49600659, 2.09230986, 0.56473997, 0.03823943, 1.19864896
      ]);
      const output = math.resizeBilinear3D(input, [4, 5], false);

      test_util.expectArraysClose(output, [
        1.19074047, 0.91373104, 1.68596613, 0.05186744, 1.69034398, -0.15654698,
        0.7130264,  0.94193673, 0.38725394, 1.30809784, 0.9045459,  2.20486879,
        1.59434628, 0.89455694, 1.68591988, 0.26748738, 0.58103991, 1.00690198,
        0.21274668, 1.25337338, 0.6183514,  3.49600649, 1.50272655, 1.73724651,
        1.68149579, 0.69152176, 0.44905344, 1.07186723, 0.03823943, 1.19864893,
        0.6183514,  3.49600649, 1.50272655, 1.73724651, 1.68149579, 0.69152176,
        0.44905344, 1.07186723, 0.03823943, 1.19864893
      ]);
    });

    it('matches tensorflow w/ random numbers alignCorners=true', math => {
      const input = Array3D.new([2, 3, 2], [
        1.56324531, 2.13817752, 1.44398421, 1.07632684, 0.59306785, -0.36970865,
        1.62451879, 1.8367334, 1.13944798, 2.01993218, 2.01919952, 2.67524054
      ]);
      const output = math.resizeBilinear3D(input, [4, 5], true);

      test_util.expectArraysClose(output, [
        1.5632453,  2.13817763, 1.50361478, 1.60725224,  1.44398427, 1.07632685,
        1.01852608, 0.35330909, 0.59306782, -0.36970866, 1.58366978, 2.03769612,
        1.46307099, 1.71427906, 1.3424722,  1.39086199,  1.20545864, 1.01806819,
        1.06844509, 0.6452744,  1.60409427, 1.93721485,  1.42252707, 1.82130599,
        1.24096,    1.70539713, 1.3923912,  1.68282723,  1.54382229, 1.66025746,
        1.62451875, 1.83673346, 1.38198328, 1.92833281,  1.13944793, 2.01993227,
        1.57932377, 2.34758639, 2.01919961, 2.67524052
      ]);
    });
  };

  test_util.describeMathCPU('resizeBilinear3D', [tests]);
  test_util.describeMathGPU('resizeBilinear3D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
