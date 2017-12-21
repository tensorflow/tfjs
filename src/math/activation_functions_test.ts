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
import {ENV} from '../environment';
import * as test_util from '../test_util';
import * as util from '../util';
// tslint:disable-next-line:max-line-length
import {EluFunc, LeakyReluFunc, ReLUFunc, SigmoidFunc, TanHFunc} from './activation_functions';
import {Array1D} from './ndarray';

describe('Activation functions', () => {
  const math = ENV.math;

  it('Tanh output', () => {
    const x = Array1D.new([1, 3, -2, 100, -100, 0]);
    const tanH = new TanHFunc();
    const y = tanH.output(math, x);

    test_util.expectNumbersClose(y.get(0), util.tanh(x.get(0)));
    test_util.expectNumbersClose(y.get(1), util.tanh(x.get(1)));
    test_util.expectNumbersClose(y.get(2), util.tanh(x.get(2)));
    test_util.expectNumbersClose(y.get(3), 1);
    test_util.expectNumbersClose(y.get(4), -1);
    test_util.expectNumbersClose(y.get(5), 0);
  });

  it('Tanh derivative', () => {
    const x = Array1D.new([1, 3, -2, 100, -100, 0]);
    const tanH = new TanHFunc();
    const y = tanH.output(math, x);
    const dx = tanH.der(math, x, y);

    test_util.expectNumbersClose(dx.get(0), 1 - Math.pow(y.get(0), 2));
    test_util.expectNumbersClose(dx.get(1), 1 - Math.pow(y.get(1), 2));
    test_util.expectNumbersClose(dx.get(2), 1 - Math.pow(y.get(2), 2));
  });

  it('ReLU output', () => {
    const x = Array1D.new([1, 3, -2]);
    const relu = new ReLUFunc();
    const y = relu.output(math, x);

    test_util.expectNumbersClose(y.get(0), 1);
    test_util.expectNumbersClose(y.get(1), 3);
    test_util.expectNumbersClose(y.get(2), 0);
  });

  it('ReLU derivative', () => {
    const x = Array1D.new([1, 3, -2]);
    const relu = new ReLUFunc();
    const y = relu.output(math, x);
    const dx = relu.der(math, x, y);

    test_util.expectNumbersClose(dx.get(0), 1);
    test_util.expectNumbersClose(dx.get(1), 1);
    test_util.expectNumbersClose(dx.get(2), 0);
  });

  it('LeakyRelu output', () => {
    const x = Array1D.new([1, 3, -2]);
    const relu = new LeakyReluFunc(0.2);
    const y = relu.output(math, x);

    test_util.expectNumbersClose(y.get(0), 1);
    test_util.expectNumbersClose(y.get(1), 3);
    test_util.expectNumbersClose(y.get(2), -0.4);
  });

  it('LeakyRelu derivative', () => {
    const x = Array1D.new([1, 3, -2]);
    const relu = new LeakyReluFunc(0.2);
    const y = relu.output(math, x);
    const dx = relu.der(math, x, y);

    test_util.expectNumbersClose(dx.get(0), 1);
    test_util.expectNumbersClose(dx.get(1), 1);
    test_util.expectNumbersClose(dx.get(2), 0.2);
  });

  it('Sigmoid output', () => {
    const x = Array1D.new([1, 3, -2, 100, -100, 0]);
    const sigmoid = new SigmoidFunc();
    const y = sigmoid.output(math, x);

    test_util.expectNumbersClose(y.get(0), 1 / (1 + Math.exp(-1)));
    test_util.expectNumbersClose(y.get(1), 1 / (1 + Math.exp(-3)));
    test_util.expectNumbersClose(y.get(2), 1 / (1 + Math.exp(2)));
    test_util.expectNumbersClose(y.get(3), 1);
    test_util.expectNumbersClose(y.get(4), 0);
    test_util.expectNumbersClose(y.get(5), 0.5);
  });

  it('Sigmoid derivative', () => {
    const x = Array1D.new([1, 3, -2, 100, -100, 0]);
    const sigmoid = new SigmoidFunc();
    const y = sigmoid.output(math, x);
    const dx = sigmoid.der(math, x, y);

    test_util.expectNumbersClose(dx.get(0), y.get(0) * (1 - y.get(0)));
    test_util.expectNumbersClose(dx.get(1), y.get(1) * (1 - y.get(1)));
    test_util.expectNumbersClose(dx.get(2), y.get(2) * (1 - y.get(2)));
  });

  it('ELU output', () => {
    const x = Array1D.new([1, 3, -2]);
    const elu = new EluFunc();
    const y = elu.output(math, x);

    test_util.expectNumbersClose(y.get(0), 1);
    test_util.expectNumbersClose(y.get(1), 3);
    test_util.expectNumbersClose(y.get(2), Math.exp(-2.0) - 1.0);
  });

  it('ELU derivative', () => {
    const x = Array1D.new([1, 3, -2]);
    const elu = new EluFunc();
    const y = elu.output(math, x);
    const dx = elu.der(math, x, y);

    test_util.expectNumbersClose(dx.get(0), 1);
    test_util.expectNumbersClose(dx.get(1), 1);
    test_util.expectNumbersClose(dx.get(2), Math.exp(-2.0));
  });
});
