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

import {Array1D, Array2D, Array3D} from './ndarray';

// math.gather
{
  const tests: MathTests = it => {
    it('1D (gather)', math => {
      const t = Array1D.new([1, 2, 3]);

      const t2 = math.gather(t, Array1D.new([0, 2, 0, 1], 'int32'), 0);

      expect(t2.shape).toEqual([4]);
      test_util.expectArraysClose(t2, [1, 3, 1, 2]);
    });

    it('2D (gather)', math => {
      const t = Array2D.new([2, 2], [1, 11, 2, 22]);
      let t2 = math.gather(t, Array1D.new([1, 0, 0, 1], 'int32'), 0);
      expect(t2.shape).toEqual([4, 2]);
      test_util.expectArraysClose(t2, [2, 22, 1, 11, 1, 11, 2, 22]);

      t2 = math.gather(t, Array1D.new([1, 0, 0, 1], 'int32'), 1);
      expect(t2.shape).toEqual([2, 4]);
      test_util.expectArraysClose(t2, [11, 1, 1, 11, 22, 2, 2, 22]);
    });

    it('3D (gather)', math => {
      const t = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);

      const t2 = math.gather(t, Array1D.new([1, 0, 0, 1], 'int32'), 2);

      expect(t2.shape).toEqual([2, 2, 4]);
      test_util.expectArraysClose(t2, 
          [2, 1, 1, 2, 4, 3, 3, 4, 6, 5, 5, 6, 8, 7, 7 ,8]);
    });

    it('bool (gather)', math => {
      const t = Array1D.new([true, false, true], 'bool');

      const t2 = math.gather(t, Array1D.new([0, 2, 0, 1], 'int32'), 0);

      expect(t2.shape).toEqual([4]);
      expect(t2.dtype).toBe('bool');
      expect(t2.getValues()).toEqual(new Uint8Array([1, 1, 1, 0]));
    });

    it('int32 (gather)', math => {
      const t = Array1D.new([1, 2, 5], 'int32');

      const t2 = math.gather(t, Array1D.new([0, 2, 0, 1], 'int32'), 0);

      expect(t2.shape).toEqual([4]);
      expect(t2.dtype).toBe('int32');
      expect(t2.getValues()).toEqual(new Int32Array([1, 5, 1, 2]));
    });

    it('propagates NaNs', math => {
      const t = Array1D.new([1, 2, NaN]);

      const t2 = math.gather(t, Array1D.new([0, 2, 0, 1], 'int32'), 0);

      expect(t2.shape).toEqual([4]);
      test_util.expectArraysClose(t2, [1, NaN, 1, 2]);
    });
  };

  test_util.describeMathCPU('gather', [tests]);
  test_util.describeMathGPU('gather', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
