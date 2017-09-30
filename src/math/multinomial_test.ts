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

import {NDArrayMath} from './math';
import {NDArrayMathCPU} from './math_cpu';
import {NDArrayMathGPU} from './math_gpu';
import {Array1D} from './ndarray';

function executeTests(mathFactory: () => NDArrayMath) {
  let math: NDArrayMath;

  beforeEach(() => {
    math = mathFactory();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('Flip a fair coin and check bounds', () => {
    const probs = Array1D.new([0.5, 0.5]);
    const result = math.multinomial(probs, 100);
    expect(result.shape).toEqual([100]);
    const [min, max] = getBounds(result.getValues());
    expect(min >= 0 - 1e-4);
    expect(max <= 1 + 1e-4);
  });

  it('Flip a two-sided coin with 100% of heads', () => {
    const probs = Array1D.new([1, 0]);
    const result = math.multinomial(probs, 100);
    expect(result.shape).toEqual([100]);
    const [min, max] = getBounds(result.getValues());
    expect(min).toBeCloseTo(0, 1e-4);
    expect(max).toBeCloseTo(0, 1e-4);
  });

  it('Flip a two-sided coin with 100% of tails', () => {
    const probs = Array1D.new([0, 1]);
    const result = math.multinomial(probs, 100);
    expect(result.shape).toEqual([100]);
    const [min, max] = getBounds(result.getValues());
    expect(min).toBeCloseTo(1, 1e-4);
    expect(max).toBeCloseTo(1, 1e-4);
  });

  it('Flip a single-sided coin throws error', () => {
    const probs = Array1D.new([1]);
    expect(() => math.multinomial(probs, 100)).toThrowError();
  });

  it('Flip a ten-sided coin and check bounds', () => {
    const n = 10;
    const probs = Array1D.zeros([n]);
    for (let i = 0; i < n; ++i) {
      probs.set(1 / n, i);
    }
    const result = math.multinomial(probs, 100);
    expect(result.shape).toEqual([100]);
    const [min, max] = getBounds(result.getValues());
    expect(min >= 0 - 1e-4);
    expect(max <= 9 + 1e-4);
  });

  function getBounds(a: Float32Array) {
    let min = Number.MAX_VALUE;
    let max = Number.MIN_VALUE;

    for (let i = 0; i < a.length; ++i) {
      min = Math.min(min, a[i]);
      max = Math.max(max, a[i]);
    }
    return [min, max];
  }
}

describe('mathCPU multinomial', () => {
  executeTests(() => new NDArrayMathCPU());
});

describe('mathGPU multinomial', () => {
  executeTests(() => new NDArrayMathGPU());
});
