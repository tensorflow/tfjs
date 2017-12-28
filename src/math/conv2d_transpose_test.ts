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

import {Array2D, Array3D, Array4D} from './ndarray';

// math.conv2dTranspose
{
  const tests: MathTests = it => {
    it('input=2x2x1,d2=1,f=2,s=1,p=0', math => {
      const origInputDepth = 1;
      const origOutputDepth = 1;
      const inputShape: [number, number, number] = [1, 1, origOutputDepth];
      const fSize = 2;
      const origPad = 0;
      const origStride = 1;

      const x = Array3D.new(inputShape, [2]);
      const w = Array4D.new(
          [fSize, fSize, origInputDepth, origOutputDepth], [3, 1, 5, 0]);

      const result = math.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad);
      const expected = [6, 2, 10, 0];

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(result, expected);
    });

    it('input=2x2x1,d2=1,f=2,s=1,p=0, batch=2', math => {
      const origInputDepth = 1;
      const origOutputDepth = 1;
      const inputShape: [number, number, number, number] =
          [2, 1, 1, origOutputDepth];
      const fSize = 2;
      const origPad = 0;
      const origStride = 1;

      const x = Array4D.new(inputShape, [2, 3]);
      const w = Array4D.new(
          [fSize, fSize, origInputDepth, origOutputDepth], [3, 1, 5, 0]);

      const result =
          math.conv2dTranspose(x, w, [2, 2, 2, 1], origStride, origPad);
      const expected = [6, 2, 10, 0, 9, 3, 15, 0];

      expect(result.shape).toEqual([2, 2, 2, 1]);
      test_util.expectArraysClose(result, expected);
    });

    it('throws when x is not rank 3', math => {
      const origInputDepth = 1;
      const origOutputDepth = 1;
      const fSize = 2;
      const origPad = 0;
      const origStride = 1;

      // tslint:disable-next-line:no-any
      const x: any = Array2D.new([2, 1], [2, 2]);
      const w = Array4D.new(
          [fSize, fSize, origInputDepth, origOutputDepth], [3, 1, 5, 0]);

      expect(() => math.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad))
          .toThrowError();
    });

    it('throws when weights is not rank 4', math => {
      const origInputDepth = 1;
      const origOutputDepth = 1;
      const inputShape: [number, number, number] = [1, 1, origOutputDepth];
      const fSize = 2;
      const origPad = 0;
      const origStride = 1;

      const x = Array3D.new(inputShape, [2]);
      // tslint:disable-next-line:no-any
      const w: any = Array3D.new([fSize, fSize, origInputDepth], [3, 1, 5, 0]);

      expect(() => math.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad))
          .toThrowError();
    });

    it('throws when x depth does not match weights original output depth',
       math => {
         const origInputDepth = 1;
         const origOutputDepth = 2;
         const wrongOrigOutputDepth = 3;
         const inputShape: [number, number, number] = [1, 1, origOutputDepth];
         const fSize = 2;
         const origPad = 0;
         const origStride = 1;

         const x = Array3D.new(inputShape, [2, 2]);
         const w = Array4D.randNormal(
             [fSize, fSize, origInputDepth, wrongOrigOutputDepth]);

         expect(
             () => math.conv2dTranspose(x, w, [2, 2, 2], origStride, origPad))
             .toThrowError();
       });
  };

  test_util.describeMathCPU('conv2dTranspose', [tests]);
  test_util.describeMathGPU('conv2dTranspose', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
