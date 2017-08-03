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

import * as util from '../util';

import {ReLUFunc, SigmoidFunc, TanHFunc} from './activation_functions';
import {NDArrayMathCPU} from './math_cpu';
import {Array1D} from './ndarray';

describe('Activation functions', () => {
  let math: NDArrayMathCPU;

  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('Tanh output', () => {
    const x = Array1D.new([1, 3, -2, 100, -100, 0]);
    const tanH = new TanHFunc();
    const y = tanH.output(math, x);

    expect(y.get(0)).toBeCloseTo(util.tanh(x.get(0)));
    expect(y.get(1)).toBeCloseTo(util.tanh(x.get(1)));
    expect(y.get(2)).toBeCloseTo(util.tanh(x.get(2)));
    expect(y.get(3)).toBeCloseTo(1);
    expect(y.get(4)).toBeCloseTo(-1);
    expect(y.get(5)).toBeCloseTo(0);
  });

  it('Tanh derivative', () => {
    const x = Array1D.new([1, 3, -2, 100, -100, 0]);
    const tanH = new TanHFunc();
    const y = tanH.output(math, x);
    const dx = tanH.der(math, x, y);

    expect(dx.get(0)).toBeCloseTo(1 - Math.pow(y.get(0), 2));
    expect(dx.get(1)).toBeCloseTo(1 - Math.pow(y.get(1), 2));
    expect(dx.get(2)).toBeCloseTo(1 - Math.pow(y.get(2), 2));
  });

  it('ReLU output', () => {
    const x = Array1D.new([1, 3, -2]);
    const relu = new ReLUFunc();
    const y = relu.output(math, x);

    expect(y.get(0)).toBeCloseTo(1);
    expect(y.get(1)).toBeCloseTo(3);
    expect(y.get(2)).toBeCloseTo(0);
  });

  it('ReLU derivative', () => {
    const x = Array1D.new([1, 3, -2]);
    const relu = new ReLUFunc();
    const y = relu.output(math, x);
    const dx = relu.der(math, x, y);

    expect(dx.get(0)).toBeCloseTo(1);
    expect(dx.get(1)).toBeCloseTo(1);
    expect(dx.get(2)).toBeCloseTo(0);
  });

  it('Sigmoid output', () => {
    const x = Array1D.new([1, 3, -2, 100, -100, 0]);
    const sigmoid = new SigmoidFunc();
    const y = sigmoid.output(math, x);

    expect(y.get(0)).toBeCloseTo(1 / (1 + Math.exp(-1)));
    expect(y.get(1)).toBeCloseTo(1 / (1 + Math.exp(-3)));
    expect(y.get(2)).toBeCloseTo(1 / (1 + Math.exp(2)));
    expect(y.get(3)).toBeCloseTo(1);
    expect(y.get(4)).toBeCloseTo(0);
    expect(y.get(5)).toBeCloseTo(0.5);
  });

  it('Sigmoid derivative', () => {
    const x = Array1D.new([1, 3, -2, 100, -100, 0]);
    const sigmoid = new SigmoidFunc();
    const y = sigmoid.output(math, x);
    const dx = sigmoid.der(math, x, y);

    expect(dx.get(0)).toBeCloseTo(y.get(0) * (1 - y.get(0)));
    expect(dx.get(1)).toBeCloseTo(y.get(1) * (1 - y.get(1)));
    expect(dx.get(2)).toBeCloseTo(y.get(2) * (1 - y.get(2)));
  });
});
