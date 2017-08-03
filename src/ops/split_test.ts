/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {Tensor} from '../graph';
import {NDArrayMathCPU} from '../math/math_cpu';
import {Array1D, Scalar} from '../math/ndarray';
import {TensorArrayMap} from '../tensor_array_map';
import * as test_util from '../test_util';

import {Split} from './split';

describe('Split operation', () => {
  let math: NDArrayMathCPU;

  let splitOp: Split;
  let tensorArrayMap: TensorArrayMap;

  beforeEach(() => {
    math = new NDArrayMathCPU();
    tensorArrayMap = new TensorArrayMap();
  });

  afterEach(() => {
    splitOp.dispose();
    tensorArrayMap.dispose();
  });

  it('Forward prop split', () => {
    const xVal = Scalar.new(-3);
    const x = new Tensor(xVal.shape);
    const y1 = new Tensor(x.shape);
    const y2 = new Tensor(x.shape);
    tensorArrayMap.set(x, xVal);
    splitOp = new Split(x, [y1, y2]);
    splitOp.feedForward(math, tensorArrayMap);
    const y1Val = tensorArrayMap.get(y1);
    const y2Val = tensorArrayMap.get(y2);
    test_util.expectArraysClose(y1Val.getValues(), xVal.getValues(), 1e-5);
    test_util.expectArraysClose(y2Val.getValues(), xVal.getValues(), 1e-5);
  });

  it('Forward+backward prop split', () => {
    const xVal = Array1D.new([4, 5, -6]);
    const x = new Tensor(xVal.shape);
    const y1 = new Tensor(x.shape);
    const y2 = new Tensor(x.shape);
    tensorArrayMap.set(x, xVal);
    splitOp = new Split(x, [y1, y2]);
    splitOp.feedForward(math, tensorArrayMap);
    const y1Val = tensorArrayMap.get(y1);
    const y2Val = tensorArrayMap.get(y2);
    test_util.expectArraysClose(y1Val.getValues(), xVal.getValues(), 1e-5);
    test_util.expectArraysClose(y2Val.getValues(), xVal.getValues(), 1e-5);

    const gradientArrayMap = new TensorArrayMap();
    gradientArrayMap.set(y1, Array1D.new([-1, 4, 3]));
    gradientArrayMap.set(y2, Array1D.new([-2, 2, -3]));
    splitOp.backProp(math, tensorArrayMap, gradientArrayMap);
    const dx = gradientArrayMap.get(x);
    const expected = new Float32Array([-3, 6, 0]);
    test_util.expectArraysClose(dx.getValues(), expected, 1e-5);
    gradientArrayMap.dispose();
  });
});
