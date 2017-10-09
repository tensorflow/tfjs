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

// math.conv2dDerWeights
{
  const tests: MathTests = it => {
    it('input=3x3x1,d2=1,f=2,s=1,p=0', math => {
      const inputDepth = 1;
      const outputDepth = 1;
      const inputShape: [number, number, number] = [3, 3, inputDepth];
      const fSize = 2;
      const stride = 1;
      const pad = 0;

      const weightsShape: [number, number, number, number] =
          [fSize, fSize, inputDepth, outputDepth];

      const x = Array3D.new(inputShape, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const dy = Array3D.new([2, 2, 1], [3, 1, 2, 0]);

      const result = math.conv2dDerFilter(x, dy, weightsShape, stride, pad);
      const expected = new Float32Array([13, 19, 31, 37]);

      expect(result.shape).toEqual(weightsShape);
      // TODO(nsthorat): Fix the precision for byte textures.
      test_util.expectArraysClose(result.getValues(), expected, 1e-1);

      x.dispose();
      dy.dispose();
    });
  };

  test_util.describeMathCPU('conv2dDerWeights', [tests]);
  test_util.describeMathGPU('conv2dDerWeights', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.conv2dDerBias
{
  const tests: MathTests = it => {
    it(' dy=2x2x2', math => {
      const outputDepth = 2;
      const dyShape: [number, number, number] = [2, 2, outputDepth];
      const dy = Array3D.new(dyShape, [1, 2, 3, 4, 5, 6, 7, 8]);

      const result = math.conv2dDerBias(dy);
      const expected = new Float32Array([16, 20]);

      expect(result.shape).toEqual([outputDepth]);
      test_util.expectArraysClose(result.getValues(), expected);

      dy.dispose();
    });
  };

  test_util.describeMathCPU('conv2dDerBias', [tests]);
  test_util.describeMathGPU('conv2dDerBias', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
