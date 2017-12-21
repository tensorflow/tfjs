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
import * as test_util from '../../test_util';
import {Tensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';
import {Log} from './log';

describe('log operation', () => {
  const math = ENV.math;

  let xTensor: Tensor;
  let yTensor: Tensor;
  let logOp: Log;
  let activations: TensorArrayMap;
  let gradients: SummedTensorArrayMap;

  beforeEach(() => {
    activations = new TensorArrayMap();
    gradients = new SummedTensorArrayMap(math);
  });

  afterEach(() => {
    activations.disposeArray(xTensor);
    activations.disposeArray(yTensor);
    gradients.disposeArray(xTensor);
    gradients.disposeArray(yTensor);
  });

  it('simple log', () => {
    const x = Array1D.new([1, 2, 3]);

    xTensor = new Tensor(x.shape);
    yTensor = new Tensor(x.shape);

    activations.set(xTensor, x);

    logOp = new Log(xTensor, yTensor);
    logOp.feedForward(math, activations);
    const y = activations.get(yTensor);

    expect(y.shape).toEqual([3]);
    test_util.expectNumbersClose(y.get(0), Math.log(x.get(0)));
    test_util.expectNumbersClose(y.get(1), Math.log(x.get(1)));
    test_util.expectNumbersClose(y.get(2), Math.log(x.get(2)));

    const dy = Array1D.new([1, 2, 3]);
    gradients.add(yTensor, dy);

    logOp.backProp(math, activations, gradients);

    const dx = gradients.get(xTensor);

    expect(dx.shape).toEqual(dx.shape);
    test_util.expectNumbersClose(dx.get(0), dy.get(0) / x.get(0));
    test_util.expectNumbersClose(dx.get(1), dy.get(1) / x.get(1));
    test_util.expectNumbersClose(dx.get(2), dy.get(2) / x.get(2));
  });
});
