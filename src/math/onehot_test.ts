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

  it('Depth 1 throws error', () => {
    const indices = Array1D.new([0, 0, 0]);
    expect(() => math.oneHot(indices, 1)).toThrowError();
  });

  it('Depth 2, diagonal', () => {
    const indices = Array1D.new([0, 1]);
    const res = math.oneHot(indices, 2);
    const expected = new Float32Array([1, 0, 0, 1]);
    expect(res.shape).toEqual([2, 2]);
    test_util.expectArraysClose(res.getValues(), expected);
  });

  it('Depth 2, transposed diagonal', () => {
    const indices = Array1D.new([1, 0]);
    const res = math.oneHot(indices, 2);
    const expected = new Float32Array([0, 1, 1, 0]);
    expect(res.shape).toEqual([2, 2]);
    test_util.expectArraysClose(res.getValues(), expected);
  });

  it('Depth 3, 4 events', () => {
    const indices = Array1D.new([2, 1, 2, 0]);
    const res = math.oneHot(indices, 3);
    const expected = new Float32Array([0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]);
    expect(res.shape).toEqual([4, 3]);
    test_util.expectArraysClose(res.getValues(), expected);
  });

  it('Depth 2 onValue=3, offValue=-2', () => {
    const indices = Array1D.new([0, 1]);
    const res = math.oneHot(indices, 2, 3, -2);
    const expected = new Float32Array([3, -2, -2, 3]);
    expect(res.shape).toEqual([2, 2]);
    test_util.expectArraysClose(res.getValues(), expected);
  });
}

describe('mathCPU oneHot', () => {
  executeTests(() => new NDArrayMathCPU());
});

describe('mathGPU oneHot', () => {
  executeTests(() => new NDArrayMathGPU());
});
