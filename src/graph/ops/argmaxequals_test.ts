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
import {Array1D} from '../../math/ndarray';
import {Tensor} from '../graph';
import {TensorArrayMap} from '../tensor_array_map';
import {ArgMaxEquals} from './argmaxequals';

describe('Argmax equals oper', () => {
  const math = ENV.math;

  let t1: Tensor;
  let t2: Tensor;
  let y: Tensor;
  let argmaxEqualsOp: ArgMaxEquals;
  let tensorArrayMap: TensorArrayMap;

  beforeEach(() => {
    tensorArrayMap = new TensorArrayMap();
  });

  afterEach(() => {
    tensorArrayMap.disposeArray(t1);
    tensorArrayMap.disposeArray(t2);
    tensorArrayMap.disposeArray(y);
  });

  it('argmax equals', () => {
    const x1 = Array1D.new([0, 2, 1]);
    const x2 = Array1D.new([2, 4, 3]);

    t1 = new Tensor(x1.shape);
    t2 = new Tensor(x2.shape);
    y = new Tensor(x1.shape);

    tensorArrayMap.set(t1, x1);
    tensorArrayMap.set(t2, x2);

    argmaxEqualsOp = new ArgMaxEquals(t1, t2, y);
    argmaxEqualsOp.feedForward(math, tensorArrayMap);
    const yVal = tensorArrayMap.get(y);

    expect(yVal.shape).toEqual([]);
    expect(yVal.dataSync()).toEqual(new Uint8Array([1]));
  });

  it('argmax not equals', () => {
    const x1 = Array1D.new([0, 2, 1]);
    const x2 = Array1D.new([5, 4, 3]);

    t1 = new Tensor(x1.shape);
    t2 = new Tensor(x2.shape);
    y = new Tensor(x1.shape);

    tensorArrayMap.set(t1, x1);
    tensorArrayMap.set(t2, x2);

    argmaxEqualsOp = new ArgMaxEquals(t1, t2, y);
    argmaxEqualsOp.feedForward(math, tensorArrayMap);
    const yVal = tensorArrayMap.get(y);

    expect(yVal.shape).toEqual([]);
    expect(yVal.dataSync()).toEqual(new Uint8Array([0]));
  });
});
