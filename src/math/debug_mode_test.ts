/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as dl from '../index';
import * as test_util from '../test_util';
import {MathTests} from '../test_util';
import * as util from '../util';

// debug mode
{
  const tests: MathTests = it => {
    it('debug mode does not error when no nans', () => {
      const a = dl.tensor1d([2, -1, 0, 3]);
      const res = dl.relu(a);
      test_util.expectArraysClose(res, [2, 0, 0, 3]);
    });

    it('debug mode errors when there are nans, float32', () => {
      const a = dl.tensor1d([2, NaN]);
      const f = () => dl.relu(a);
      expect(f).toThrowError();
    });

    it('debug mode errors when there are nans, int32', () => {
      const a = dl.tensor1d([2, util.NAN_INT32], 'int32');
      const f = () => dl.relu(a);
      expect(f).toThrowError();
    });

    it('debug mode errors when there are nans, bool', () => {
      const a = dl.tensor1d([1, util.NAN_BOOL], 'bool');
      const f = () => dl.relu(a);
      expect(f).toThrowError();
    });

    it('A x B', () => {
      const a = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
      const b = dl.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

      const c = dl.matMul(a, b);

      expect(c.shape).toEqual([2, 2]);
      test_util.expectArraysClose(c, [0, 8, -3, 20]);
    });
  };

  test_util.describeMathCPU('debug mode on', [tests], [{'DEBUG': true}]);
  test_util.describeMathGPU('debug mode on ', [tests], [
    {'DEBUG': true, 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'DEBUG': true, 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'DEBUG': true, 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// debug mode off
{
  const gpuTests: MathTests = it => {
    it('no errors where there are nans, and debug mode is disabled', () => {
      const a = dl.tensor1d([2, NaN]);
      const res = dl.relu(a);
      test_util.expectArraysClose(res, [2, NaN]);
    });
  };

  test_util.describeMathCPU('debug mode off', [gpuTests], [{'DEBUG': false}]);
  test_util.describeMathGPU('debug mode off', [gpuTests], [
    {'DEBUG': false, 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'DEBUG': false, 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2}, {
      'DEBUG': false,
      'WEBGL_FLOAT_TEXTURE_ENABLED': false,
      'WEBGL_VERSION': 1
    }
  ]);
}
