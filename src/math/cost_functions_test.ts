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

import {SquareCostFunc} from './cost_functions';
import {NDArrayMathCPU} from './math_cpu';
import {Array1D} from './ndarray';

describe('Cost functions', () => {
  let math: NDArrayMathCPU;

  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('Square cost', () => {
    const y = Array1D.new([1, 3, -2]);
    const target = Array1D.new([0, 3, -1.5]);
    const square = new SquareCostFunc();
    const cost = square.cost(math, y, target);

    // The cost function is 1/2 * (y - target)^2
    expect(cost.get(0)).toBeCloseTo(1 / 2);
    expect(cost.get(1)).toBeCloseTo(0 / 2);
    expect(cost.get(2)).toBeCloseTo(0.25 / 2);
  });

  it('Square derivative', () => {
    const y = Array1D.new([1, 3, -2]);
    const target = Array1D.new([0, 3, -1.5]);
    const square = new SquareCostFunc();
    const dy = square.der(math, y, target);

    expect(dy.get(0)).toBeCloseTo(1);
    expect(dy.get(1)).toBeCloseTo(0);
    expect(dy.get(2)).toBeCloseTo(-0.5);
  });
});  // Close describe.
