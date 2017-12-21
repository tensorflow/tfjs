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
import {Array1D, Array2D} from '../../math/ndarray';
import * as test_util from '../../test_util';
import {Tensor} from '../graph';
import {TensorArrayMap} from '../tensor_array_map';

import {ArgMax} from './argmax';

describe('Argmax oper', () => {
  const math = ENV.math;

  let x: Tensor;
  let y: Tensor;
  let tensorArrayMap: TensorArrayMap;

  beforeEach(() => {
    tensorArrayMap = new TensorArrayMap();
  });

  afterEach(() => {
    tensorArrayMap.disposeArray(x);
    tensorArrayMap.disposeArray(y);
  });

  it('argmax of Array1D', () => {
    const vals = Array1D.new([0, 2, 1]);
    x = new Tensor(vals.shape);
    y = new Tensor([]);
    tensorArrayMap.set(x, vals);

    const argmaxOp = new ArgMax(x, y);
    argmaxOp.feedForward(math, tensorArrayMap);
    const yVal = tensorArrayMap.get(y);

    expect(yVal.shape).toEqual([]);
    test_util.expectNumbersClose(yVal.get(), 1);
  });

  it('argmax of Array2D', () => {
    const vals = Array2D.new([2, 3], [[0, 2, 1], [2, 3, 0]]);
    x = new Tensor(vals.shape);
    y = new Tensor([]);
    tensorArrayMap.set(x, vals);

    const argmaxOp = new ArgMax(x, y);
    argmaxOp.feedForward(math, tensorArrayMap);
    const yVal = tensorArrayMap.get(y);

    expect(yVal.shape).toEqual([]);
    test_util.expectNumbersClose(yVal.get(), 4);
  });
});
