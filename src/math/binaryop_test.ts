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

// math.prelu
{
  const tests: MathTests = it => {
    it('basic', math => {
      const x = Array1D.new([0, 1, -2, -4]);
      const a = Array1D.new([0.15, 0.2, 0.25, 0.15]);
      const result = math.prelu(x, a);

      expect(result.shape).toEqual(x.shape);
      test_util.expectArraysClose(result, [0, 1, -0.5, -0.6]);
    });

    it('propagates NaN', math => {
      const x = Array1D.new([0, 1, NaN]);
      const a = Array1D.new([0.15, 0.2, 0.25]);
      const result = math.prelu(x, a);

      expect(result.shape).toEqual(x.shape);
      test_util.expectArraysClose(result, [0, 1, NaN]);
    });
  };

  test_util.describeMathCPU('prelu', [tests]);
  test_util.describeMathGPU('prelu', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.preluDer
{
  const tests: MathTests = it => {
    it('basic', math => {
      const x = Array1D.new([0.5, 3, -0.1, -4]);
      const a = Array1D.new([0.2, 0.4, 0.25, 0.15]);
      const result = math.preluDer(x, a);

      expect(result.shape).toEqual(x.shape);
      test_util.expectArraysClose(result, [1, 1, 0.25, 0.15]);
    });

    it('propagates NaN', math => {
      const x = Array1D.new([0.5, -0.1, NaN]);
      const a = Array1D.new([0.2, 0.3, 0.25]);
      const result = math.preluDer(x, a);

      expect(result.shape).toEqual(x.shape);
      test_util.expectArraysClose(result, [1, 0.3, NaN]);
    });
  };

  test_util.describeMathCPU('preluDer', [tests]);
  test_util.describeMathGPU('preluDer', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
