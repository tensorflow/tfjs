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

import {InputProvider} from '../data/input_provider';
import {ENV} from '../environment';
import {NDArrayMath} from '../math/math';
import {Array1D, NDArray, Scalar} from '../math/ndarray';
import {SGDOptimizer} from '../math/optimizers/sgd_optimizer';
import * as test_util from '../test_util';

import {Graph, Tensor} from './graph';
import {FeedDictionary, FeedEntry, Session} from './session';

describe('FeedDictionary', () => {
  it('ctor leaves dict empty if no args are passed', () => {
    expect(Object.keys(new FeedDictionary().dict).length).toEqual(0);
  });

  it('ctor populates dict from only feed entry', () => {
    const math = ENV.math;
    math.scope(() => {
      const e: FeedEntry = {tensor: new Tensor([]), data: NDArray.zeros([1])};
      const d = new FeedDictionary([e]);
      expect(Object.keys(d.dict).length).toEqual(1);
      expect(d.dict[e.tensor.id]).toBe(e);
    });
  });

  it('ctor populates dict from many entries', () => {
    const entries: FeedEntry[] = [
      {tensor: new Tensor([]), data: NDArray.zeros([1])},
      {tensor: new Tensor([]), data: NDArray.zeros([1])},
      {tensor: new Tensor([]), data: NDArray.zeros([1])},
      {tensor: new Tensor([]), data: NDArray.zeros([1])}
    ];
    const d = new FeedDictionary(entries);
    expect(Object.keys(d.dict).length).toEqual(entries.length);
    entries.forEach(entry => expect(d.dict[entry.tensor.id]).toBe(entry));
  });

  it('add adds entry to map keyed on tensor id', () => {
    const t = new Tensor([]);
    const nda = NDArray.zeros([1]);
    const fd = new FeedDictionary([{tensor: t, data: nda}]);
    expect(fd.dict[t.id].tensor).toBe(t);
    expect(fd.dict[t.id].data).toBe(nda);
  });
});

describe('Session', () => {
  let g: Graph;

  beforeEach(() => g = new Graph());

  it('mnist fc', () => {
    const math = ENV.math;
    const input = g.placeholder('input', [28 * 28]);
    const fc0W = g.variable('fc0W', NDArray.zeros([32, 28 * 28]));
    const fc0B = g.variable('fc0B', NDArray.zeros([32]));
    const fc0 = g.add(g.matmul(fc0W, input), fc0B);
    const relu0 = g.relu(fc0);
    const fc1W = g.variable('fc1W', NDArray.zeros([32, 32]));
    const fc1B = g.variable('fc1B', NDArray.zeros([32]));
    const fc1 = g.add(g.matmul(fc1W, relu0), fc1B);
    const relu1 = g.relu(fc1);
    const fc2W = g.variable('fc2W', NDArray.zeros([32, 32]));
    const fc2B = g.variable('fc2B', NDArray.zeros([32]));
    const fc2 = g.add(g.matmul(fc2W, relu1), fc2B);
    const relu2 = g.relu(fc2);
    const fc3W = g.variable('fc3W', NDArray.zeros([10, 32]));
    const fc3B = g.variable('fc3B', NDArray.zeros([10]));
    const fc3 = g.add(g.matmul(fc3W, relu2), fc3B);

    const session = new Session(g, math);
    session.eval(fc3, [{tensor: input, data: NDArray.zeros([28 * 28])}]);
  });

  it('y=x^2 + 3: CPU', () => {
    const x = g.placeholder('x', [2]);
    const y = g.add(g.square(x), g.constant(3));
    const session = new Session(g, ENV.math);
    const yVal = session.eval(y, [{tensor: x, data: Array1D.new([5, 4])}]);
    const expected = new Float32Array([28, 19]);
    test_util.expectArraysClose(yVal.dataSync(), expected);
  });

  it('y=x^2 + 3: GPU', () => {
    const math = ENV.math;
    const x = g.placeholder('x', [2]);
    const y = g.add(g.square(x), g.constant(3));
    const session = new Session(g, math);

    math.scope(() => {
      const yVal = session.eval(y, [{tensor: x, data: Array1D.new([5, 4])}]);
      const expected = new Float32Array([28, 19]);
      test_util.expectArraysClose(yVal.dataSync(), expected);
    });
  });

  it('Non-placeholder feed: y=x^2 + 3 (feed x^2)', () => {
    const math = ENV.math;
    const x = g.placeholder('x', [2]);
    const xSquared = g.square(x);
    const y = g.add(xSquared, g.constant(3));
    const session = new Session(g, math);

    math.scope(() => {
      const yVal =
          session.eval(y, [{tensor: xSquared, data: Array1D.new([25, 16])}]);
      const expected = new Float32Array([28, 19]);
      test_util.expectArraysClose(yVal.dataSync(), expected);
    });
  });

  it('Eval multiple tensors that share graph: y=x^2 + 3, z=x^2 + 2', () => {
    const math = ENV.math;
    const x = g.placeholder('x', [2]);
    const xSquared = g.square(x);
    const y = g.add(xSquared, g.constant(3));
    const z = g.add(xSquared, g.constant(2));
    const session = new Session(g, math);

    math.scope(() => {
      const result =
          session.evalAll([y, z], [{tensor: x, data: Array1D.new([5, 4])}]);
      const expectedY = new Float32Array([28, 19]);
      const expectedZ = new Float32Array([27, 18]);
      test_util.expectArraysClose(result[0].dataSync(), expectedY);
      test_util.expectArraysClose(result[1].dataSync(), expectedZ);
    });
  });

  it('Eval 2 tensors that share a split graph: y=x^2 + x, z=y + 1', () => {
    const math = ENV.math;
    const x = g.placeholder('x', [2]);
    const xSquared = g.square(x);
    const y = g.add(xSquared, x);
    const z = g.add(y, g.constant(1));
    const session = new Session(g, math);

    math.scope(() => {
      const result1 = session.eval(y, [{tensor: x, data: Array1D.new([5, 4])}]);
      const expectedY = new Float32Array([30, 20]);
      test_util.expectArraysClose(result1.dataSync(), expectedY);

      const result2 = session.eval(z, [{tensor: x, data: Array1D.new([5, 4])}]);
      const expectedZ = new Float32Array([31, 21]);
      test_util.expectArraysClose(result2.dataSync(), expectedZ);
    });
  });

  it('Backprop through a  with 2 outputs, input is scalar', () => {
    const math = ENV.math;
    const two = Scalar.new(2);
    const one = Scalar.new(1);
    const negOne = Scalar.new(-1);

    const x = g.placeholder('x', []);
    const y = g.square(x);
    const z = g.add(x, g.constant(3));
    const w = g.add(y, z);

    const optimizer = new SGDOptimizer(0.1);
    const session = new Session(g, math);
    let idx = 0;
    const xs: Scalar[] = [two, one, negOne];
    const inputProvider: InputProvider = {
      getNextCopy() {
        return xs[idx++];
      },
      disposeCopy(math, example) {}
    };

    // w = x^2 + x + 3
    // dw/dx = 2x + 1
    session.train(w, [{tensor: x, data: inputProvider}], 1, optimizer);
    let dwdx = session.gradientArrayMap.get(x).get();
    expect(dwdx).toBe(5);

    session.train(w, [{tensor: x, data: inputProvider}], 1, optimizer);
    dwdx = session.gradientArrayMap.get(x).get();
    expect(dwdx).toBe(3);

    session.train(w, [{tensor: x, data: inputProvider}], 1, optimizer);
    dwdx = session.gradientArrayMap.get(x).get();
    expect(dwdx).toBe(-1);
  });

  it('Backprop through a node with 2 outputs, input is Array1D', () => {
    const math = ENV.math;
    const x = g.placeholder('x', [2]);
    const y = g.square(x);
    const z = g.add(x, g.constant(3));
    const w = g.reduceSum(g.add(y, z));

    const optimizer = new SGDOptimizer(0.1);
    const session = new Session(g, math);
    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([2, 4]);
      },
      disposeCopy(math, example) {}
    };

    // w = reduce_sum(x^2 + x + 3)
    // dw/dx = [2*x_1 + 1, 2*x_2 + 1]
    session.train(w, [{tensor: x, data: inputProvider}], 1, optimizer);
    const dwdx = session.gradientArrayMap.get(x).dataSync();
    test_util.expectArraysClose(dwdx, new Float32Array([5, 9]));
  });

  it('Specify which variables to update (var_list)', () => {
    const math = ENV.math;
    const x = g.placeholder('x', [2]);
    const b0 = g.variable('b0', NDArray.zeros([2]));
    const p = g.add(x, b0);
    const q = g.square(p);
    const b1 = g.variable('b1', NDArray.zeros([2]));
    const r = g.add(q, b1);
    const yPrediction = g.reduceSum(r);
    const yTrue = g.constant(1);
    const cost = g.meanSquaredCost(yTrue, yPrediction);

    const session = new Session(g, math);
    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([1, 2]);
      },
      disposeCopy(math, example) {}
    };

    // prediction = reduce_sum((x + b0)^2 + b1)
    // dE/db0 = (1 - prediction) * [- 2*x_1 - 2*b0_1, - 2*x_2 - 2*b0_2]
    // dE/db0_{x=[1,2], b0=[0,0]} = (8, 16)

    // Update only b0
    const optimizerOnlyB0 = new SGDOptimizer(0.1, [b0.node]);
    session.train(
        cost, [{tensor: x, data: inputProvider}], 2, optimizerOnlyB0,
        undefined);
    const b0After1 = session.activationArrayMap.get(b0).dataSync();
    const b1After1 = session.activationArrayMap.get(b1).dataSync();

    test_util.expectArraysClose(b0After1, new Float32Array([-0.8, -1.6]));
    test_util.expectArraysClose(b1After1, new Float32Array([0, 0]));

    // Update both b0 and b1
    const optimizerAll = new SGDOptimizer(0.1);
    session.train(
        cost, [{tensor: x, data: inputProvider}], 2, optimizerAll, undefined);
    const b0After2 = session.activationArrayMap.get(b0).dataSync();
    const b1After2 = session.activationArrayMap.get(b1).dataSync();

    expect(b0After2 === b0After1).toEqual(false);
    expect(b1After2 === b1After1).toEqual(false);
  });

  it('Safe mode math, no math scope eval throws', () => {
    const safeMode = true;
    const math = new NDArrayMath('webgl', safeMode);
    ENV.setMath(math);

    expect(() => {
      const x = g.placeholder('x', [2]);
      const y = g.square(x);
      const session = new Session(g, math);
      session.eval(y, [{tensor: x, data: Array1D.new([5, 4])}]);
    }).toThrowError();
    ENV.reset();
  });

  it('Safe mode math, math scope eval does not throw', () => {
    const safeMode = true;
    const math = new NDArrayMath('webgl', safeMode);
    ENV.setMath(math);

    math.scope(() => {
      const x = g.placeholder('x', [2]);
      const y = g.square(x);
      const session = new Session(g, math);
      const yVal = session.eval(y, [{tensor: x, data: Array1D.new([5, 4])}]);
      const expected = new Float32Array([25, 16]);
      test_util.expectArraysClose(yVal.dataSync(), expected);
    });
    ENV.reset();
  });

  it('Safe mode math, math scope train does not throw', () => {
    const safeMode = true;
    const math = new NDArrayMath('webgl', safeMode);
    ENV.setMath(math);

    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([2, 4]);
      },
      disposeCopy(math, example) {}
    };

    math.scope(() => {
      const optimizer = new SGDOptimizer(0.1);
      const session = new Session(g, math);
      const x = g.placeholder('x', [2]);
      const y = g.square(x);
      const z = g.add(x, g.constant(3));
      // w = reduce_sum(x^2 + x + 3)
      // dw/dx = [2*x_1 + 1, 2*x_2 + 1]
      const w = g.reduceSum(g.add(y, z));
      session.train(w, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dwdx = session.gradientArrayMap.get(x).dataSync();
      test_util.expectArraysClose(dwdx, new Float32Array([5, 9]));
    });
    ENV.reset();
  });

  it('Safe mode math, no math scope train throws', () => {
    const safeMode = true;
    const math = new NDArrayMath('webgl', safeMode);
    ENV.setMath(math);

    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([2, 4]);
      },
      disposeCopy(math, example) {}
    };

    expect(() => {
      const session = new Session(g, math);
      const optimizer = new SGDOptimizer(0.1);
      const x = g.placeholder('x', [2]);
      const y = g.square(x);
      const z = g.add(x, g.constant(3));
      const w = g.reduceSum(g.add(y, z));
      session.train(w, [{tensor: x, data: inputProvider}], 1, optimizer);
    }).toThrowError();

    ENV.reset();
  });
});
