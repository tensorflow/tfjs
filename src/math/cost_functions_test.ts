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
import {SquareCostFunc} from './cost_functions';
import {Array1D} from './ndarray';

const tests: MathTests = it => {
  it('Square cost', math => {
    const y = Array1D.new([1, 3, -2]);
    const target = Array1D.new([0, 3, -1.5]);
    const square = new SquareCostFunc();
    const cost = square.cost(math, y, target);

    // The cost function is 1/2 * (y - target)^2
    test_util.expectNumbersClose(cost.get(0), 1 / 2);
    test_util.expectNumbersClose(cost.get(1), 0 / 2);
    test_util.expectNumbersClose(cost.get(2), 0.25 / 2);
  });

  it('Square derivative', math => {
    const y = Array1D.new([1, 3, -2]);
    const target = Array1D.new([0, 3, -1.5]);
    const square = new SquareCostFunc();
    const dy = square.der(math, y, target);

    test_util.expectNumbersClose(dy.get(0), 1);
    test_util.expectNumbersClose(dy.get(1), 0);
    test_util.expectNumbersClose(dy.get(2), -0.5);
  });
};

test_util.describeMathCPU('Square cost', [tests]);
test_util.describeMathGPU('Square cost', [tests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
