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
  it('regular test', math => {
    const y = math.softmax(Array1D.new([2, 1, 3]));

    test_util.expectArraysClose(
        y.getValues(), new Float32Array([0.24472847, 0.09003057, 0.66524095]));
    test_util.expectNumbersClose(y.get(0) + y.get(1) + y.get(2), 1);
  });

  it('overflow', math => {
    const y = math.softmax(Array1D.new([1000, 1000]));

    test_util.expectArraysClose(y.getValues(), new Float32Array([0.5, 0.5]));
  });

  it('underflow', math => {
    const y = math.softmax(Array1D.new([-1000, -1000]));

    test_util.expectArraysClose(y.getValues(), new Float32Array([0.5, 0.5]));
  });

  it('Huge difference between probabilities', math => {
    const y = math.softmax(Array1D.new([-1000, +1000]));

    test_util.expectArraysClose(y.getValues(), new Float32Array([0.0, 1]));
  });

  it('Propagates NaNs', math => {
    const a = Array1D.new([2, 1, NaN]);

    const y = math.softmax(a);

    test_util.expectArraysClose(
        y.getValues(), new Float32Array([NaN, NaN, NaN]));

    a.dispose();
  });
};

test_util.describeMathCPU('softmax', [tests]);
test_util.describeMathGPU('softmax', [tests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
