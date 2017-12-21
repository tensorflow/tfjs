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

import {ENV} from '../../environment';
import {Array1D, Array2D, Scalar} from '../../math/ndarray';
import {Tensor} from '../graph';
import {TensorArrayMap} from '../tensor_array_map';

import {ReduceSum} from './reduce_sum';

describe('Reduce sum operation', () => {
  const math = ENV.math;
  let reduceSumOp: ReduceSum;
  let activations: TensorArrayMap;

  beforeEach(() => {
    activations = new TensorArrayMap();
  });

  afterEach(() => {
    reduceSumOp.dispose();
    activations.dispose();
  });

  it('Reduces a scalar', () => {
    const xVal = Scalar.new(-3);
    const x = new Tensor(xVal.shape);
    const y = new Tensor([]);

    activations.set(x, xVal);
    reduceSumOp = new ReduceSum(x, y);
    reduceSumOp.feedForward(math, activations);
    const yVal = activations.get(y);

    expect(yVal.shape).toEqual([]);
    expect(yVal.get()).toBe(-3);
  });

  it('Reduces a 1-D tensor', () => {
    const xVal = Array1D.new([1, 2, 3]);
    const x = new Tensor(xVal.shape);
    const y = new Tensor([]);

    activations.set(x, xVal);
    reduceSumOp = new ReduceSum(x, y);
    reduceSumOp.feedForward(math, activations);
    const yVal = activations.get(y);

    expect(yVal.shape).toEqual([]);
    expect(yVal.get()).toBe(6);
  });

  it('Reduces a 2-D tensor', () => {
    const xVal = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const x = new Tensor(xVal.shape);
    const y = new Tensor([]);

    activations.set(x, xVal);
    reduceSumOp = new ReduceSum(x, y);
    reduceSumOp.feedForward(math, activations);
    const yVal = activations.get(y);

    expect(yVal.shape).toEqual([]);
    expect(yVal.get()).toBe(21);
  });
});
