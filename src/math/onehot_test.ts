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

const tests: MathTests = it => {
  it('Depth 1 throws error', math => {
    const indices = Array1D.new([0, 0, 0]);
    expect(() => math.oneHot(indices, 1)).toThrowError();
  });

  it('Depth 2, diagonal', math => {
    const indices = Array1D.new([0, 1]);
    const res = math.oneHot(indices, 2);

    expect(res.shape).toEqual([2, 2]);
    test_util.expectArraysClose(res, [1, 0, 0, 1]);
  });

  it('Depth 2, transposed diagonal', math => {
    const indices = Array1D.new([1, 0]);
    const res = math.oneHot(indices, 2);

    expect(res.shape).toEqual([2, 2]);
    test_util.expectArraysClose(res, [0, 1, 1, 0]);
  });

  it('Depth 3, 4 events', math => {
    const indices = Array1D.new([2, 1, 2, 0]);
    const res = math.oneHot(indices, 3);

    expect(res.shape).toEqual([4, 3]);
    test_util.expectArraysClose(res, [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]);
  });

  it('Depth 2 onValue=3, offValue=-2', math => {
    const indices = Array1D.new([0, 1]);
    const res = math.oneHot(indices, 2, 3, -2);

    expect(res.shape).toEqual([2, 2]);
    test_util.expectArraysClose(res, [3, -2, -2, 3]);
  });
};

test_util.describeMathCPU('oneHot', [tests]);
test_util.describeMathGPU('oneHot', [tests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
