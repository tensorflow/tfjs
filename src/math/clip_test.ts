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

import {Array1D} from './ndarray';

{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array1D.new([3, -1, 0, 100, -7, 2]);
      const min = -1;
      const max = 50;

      const result = math.clip(a, min, max);

      test_util.expectArraysClose(result, [3, -1, 0, 50, -1, 2]);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([3, -1, 0, 100, -7, 2, NaN]);
      const min = -1;
      const max = 50;

      const result = math.clip(a, min, max);

      test_util.expectArraysClose(result, [3, -1, 0, 50, -1, 2, NaN]);
    });

    it('min greater than max', math => {
      const a = Array1D.new([3, -1, 0, 100, -7, 2]);
      const min = 1;
      const max = -1;

      const f = () => {
        math.clip(a, min, max);
      };
      expect(f).toThrowError();
    });
  };

  test_util.describeMathCPU('clip', [tests]);
  test_util.describeMathGPU('clip', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
