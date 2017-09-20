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
import {NDArrayMathCPU} from '../math/math_cpu';
import {NDArrayMathGPU} from '../math/math_gpu';
import {Array1D, NDArray, Scalar} from '../math/ndarray';

import {Graph, Tensor} from './graph';
import {AdagradOptimizer} from './optimizers/adagrad_optimizer';
import {MomentumOptimizer} from './optimizers/momentum_optimizer';
import {RMSPropOptimizer} from './optimizers/rmsprop_optimizer';
import {FeedDictionary, FeedEntry, Session} from './session';
import {SGDOptimizer} from './optimizers/sgd_optimizer';
import * as test_util from '../test_util';


describe('FeedDictionary', () => {
  it('ctor leaves dict empty if no args are passed', () => {
    expect(Object.keys(new FeedDictionary().dict).length).toEqual(0);
  });

  it('ctor populates dict from only feed entry', () => {
    const e: FeedEntry = {tensor: new Tensor([]), data: NDArray.zeros([1])};
    const d = new FeedDictionary([e]);
    expect(Object.keys(d.dict).length).toEqual(1);
    expect(d.dict[e.tensor.id]).toBe(e);
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

    const session = new Session(g, new NDArrayMathCPU());
    session.eval(fc3, [{tensor: input, data: NDArray.zeros([28 * 28])}]);
  });

  it('y=x^2 + 3: CPU', () => {
    const x = g.placeholder('x', [2]);
    const y = g.add(g.square(x), g.constant(3));
    const session = new Session(g, new NDArrayMathCPU());
    const yVal = session.eval(y, [{tensor: x, data: Array1D.new([5, 4])}]);
    const expected = new Float32Array([28, 19]);
    test_util.expectArraysClose(yVal.getValues(), expected, 1e-5);
  });

  it('y=x^2 + 3: GPU', () => {
    const x = g.placeholder('x', [2]);
    const y = g.add(g.square(x), g.constant(3));
    const math = new NDArrayMathGPU();
    const session = new Session(g, math);

    math.scope(() => {
      const yVal = session.eval(y, [{tensor: x, data: Array1D.new([5, 4])}]);
      const expected = new Float32Array([28, 19]);
      test_util.expectArraysClose(yVal.getValues(), expected, 1e-5);
    });
  });

  it('Non-placeholder feed: y=x^2 + 3 (feed x^2)', () => {
    const x = g.placeholder('x', [2]);
    const xSquared = g.square(x);
    const y = g.add(xSquared, g.constant(3));
    const math = new NDArrayMathGPU();
    const session = new Session(g, math);

    math.scope(() => {
      const yVal =
          session.eval(y, [{tensor: xSquared, data: Array1D.new([25, 16])}]);
      const expected = new Float32Array([28, 19]);
      test_util.expectArraysClose(yVal.getValues(), expected, 1e-5);
    });
  });

  it('Eval multiple tensors that share graph: y=x^2 + 3, z=x^2 + 2', () => {
    const x = g.placeholder('x', [2]);
    const xSquared = g.square(x);
    const y = g.add(xSquared, g.constant(3));
    const z = g.add(xSquared, g.constant(2));
    const math = new NDArrayMathGPU();
    const session = new Session(g, math);

    math.scope(() => {
      const result =
          session.evalAll([y, z], [{tensor: x, data: Array1D.new([5, 4])}]);
      const expectedY = new Float32Array([28, 19]);
      const expectedZ = new Float32Array([27, 18]);
      test_util.expectArraysClose(result[0].getValues(), expectedY, 1e-5);
      test_util.expectArraysClose(result[1].getValues(), expectedZ, 1e-5);
    });
  });

  it('Eval 2 tensors that share a split graph: y=x^2 + x, z=y + 1', () => {
    const x = g.placeholder('x', [2]);
    const xSquared = g.square(x);
    const y = g.add(xSquared, x);
    const z = g.add(y, g.constant(1));
    const math = new NDArrayMathGPU();
    const session = new Session(g, math);

    math.scope(() => {
      const result1 = session.eval(y, [{tensor: x, data: Array1D.new([5, 4])}]);
      const expectedY = new Float32Array([30, 20]);
      test_util.expectArraysClose(result1.getValues(), expectedY, 1e-5);

      const result2 = session.eval(z, [{tensor: x, data: Array1D.new([5, 4])}]);
      const expectedZ = new Float32Array([31, 21]);
      test_util.expectArraysClose(result2.getValues(), expectedZ, 1e-5);
    });
  });

  it('Backprop through a  with 2 outputs, input is scalar', () => {
    const x = g.placeholder('x', []);
    const y = g.square(x);
    const z = g.add(x, g.constant(3));
    const w = g.add(y, z);

    const optimizer = new SGDOptimizer(0.1);
    const session = new Session(g, new NDArrayMathCPU());
    let idx = 0;
    const xs: Scalar[] = [Scalar.TWO, Scalar.ONE, Scalar.NEG_ONE];
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
    const x = g.placeholder('x', [2]);
    const y = g.square(x);
    const z = g.add(x, g.constant(3));
    const w = g.reduceSum(g.add(y, z));

    const optimizer = new SGDOptimizer(0.1);
    const session = new Session(g, new NDArrayMathCPU());
    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([2, 4]);
      },
      disposeCopy(math, example) {}
    };

    // w = reduce_sum(x^2 + x + 3)
    // dw/dx = [2*x_1 + 1, 2*x_2 + 1]
    session.train(w, [{tensor: x, data: inputProvider}], 1, optimizer);
    const dwdx = session.gradientArrayMap.get(x).getValues();
    test_util.expectArraysClose(dwdx, new Float32Array([5, 9]), 1e-5);
  });

  it('Specify which variables to update (var_list)', () => {
    const x = g.placeholder('x', [2]);
    const b0 = g.variable('b0', NDArray.zeros([2]));
    const p = g.add(x, b0);
    const q = g.square(p);
    const b1 = g.variable('b1', NDArray.zeros([2]));
    const r = g.add(q, b1);
    const yPrediction = g.reduceSum(r);
    const yTrue = g.constant(1);
    const cost = g.meanSquaredCost(yTrue, yPrediction);

    const session = new Session(g, new NDArrayMathCPU());
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
    const b0After1 = session.activationArrayMap.get(b0).getValues();
    const b1After1 = session.activationArrayMap.get(b1).getValues();

    test_util.expectArraysClose(b0After1, new Float32Array([-0.8, -1.6]), 1e-5);
    test_util.expectArraysClose(b1After1, new Float32Array([0, 0]), 1e-5);

    // Update both b0 and b1
    const optimizerAll = new SGDOptimizer(0.1);
    session.train(
        cost, [{tensor: x, data: inputProvider}], 2, optimizerAll, undefined);
    const b0After2 = session.activationArrayMap.get(b0).getValues();
    const b1After2 = session.activationArrayMap.get(b1).getValues();

    expect(b0After2 === b0After1).toEqual(false);
    expect(b1After2 === b1After1).toEqual(false);
  });

  it('Safe mode math, no math scope eval throws', () => {
    const safeMode = true;
    const x = g.placeholder('x', [2]);
    const y = g.square(x);
    const math = new NDArrayMathCPU(safeMode);
    const session = new Session(g, math);

    expect(() => session.eval(y, [{tensor: x, data: Array1D.new([5, 4])}]))
        .toThrowError();
  });

  it('Safe mode math, math scope eval does not throw', () => {
    const safeMode = true;
    const x = g.placeholder('x', [2]);
    const y = g.square(x);
    const math = new NDArrayMathCPU(safeMode);
    const session = new Session(g, math);

    math.scope(() => {
      const yVal = session.eval(y, [{tensor: x, data: Array1D.new([5, 4])}]);
      const expected = new Float32Array([25, 16]);
      test_util.expectArraysClose(yVal.getValues(), expected, 1e-5);
    });
  });

  it('Safe mode math, math scope train does not throw', () => {
    const x = g.placeholder('x', [2]);
    const y = g.square(x);
    const z = g.add(x, g.constant(3));
    const w = g.reduceSum(g.add(y, z));

    const safeMode = true;
    const optimizer = new SGDOptimizer(0.1);
    const math = new NDArrayMathCPU(safeMode);
    const session = new Session(g, math);
    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([2, 4]);
      },
      disposeCopy(math, example) {}
    };

    math.scope(() => {
      // w = reduce_sum(x^2 + x + 3)
      // dw/dx = [2*x_1 + 1, 2*x_2 + 1]
      session.train(w, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dwdx = session.gradientArrayMap.get(x).getValues();
      test_util.expectArraysClose(dwdx, new Float32Array([5, 9]), 1e-5);
    });
  });

  it('Safe mode math, math scope train does not throw', () => {
    const x = g.placeholder('x', [2]);
    const w = g.variable('w', NDArray.zeros([1, 2]));
    const b = g.variable('b', NDArray.zeros([1]));
    const y = g.reduceSum(g.add(g.matmul(w, x), b));

    const safeMode = true;
    const optimizer = new MomentumOptimizer(0.1, 0.5);
    const math = new NDArrayMathCPU(safeMode);
    const session = new Session(g, math);
    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([2, 4]);
      },
      disposeCopy(math, example) {}
    };

    math.scope(() => {
      // w = reduce_sum(w_1*x_1 + w_2*x_2 + b)
      // velocity_w = [momentum* old_vel_w1 + x_1,
      //                momentum* old_vel_w2 + x_2] = [2,4]
      // w = [ w_old - lr*vel_w1, w_old - lr*vel_w2] = [-0.2, -0.4]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw = session.activationArrayMap.get(w).getValues();
      test_util.expectArraysClose(dydw, new Float32Array([-.2, -0.4]), 1e-5);
      // velocity_w = [momentum* old_vel_w1 + x_1,
      //                momentum* old_vel_w2 + x_2] = [3,6]
      // w = [ w_old - lr*vel_w1, w_old - lr*vel_w2] = [-0.5, -1.0]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw2 = session.activationArrayMap.get(w).getValues();
      test_util.expectArraysClose(dydw2, new Float32Array([-.5, -1.0]), 2e-5);
    });
  });


  it('Safe mode math, math scope train does not throw', () => {
    const x = g.placeholder('x', [2]);
    const w = g.variable('w', NDArray.zeros([1, 2]));
    const b = g.variable('b', NDArray.zeros([1]));
    const y = g.reduceSum(g.add(g.matmul(w, x), b));

    const safeMode = true;
    const optimizer = new AdagradOptimizer(0.1, 0.5);
    const math = new NDArrayMathCPU(safeMode);
    const session = new Session(g, math);
    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([2, 4]);
      },
      disposeCopy(math, example) {}
    };

    math.scope(() => {
      // w = reduce_sum(w_1*x_1 + w_2*x_2 + b)
      // cache = [old_cache_w1 + grad_w1**2,
      //                old_cache_w2 + grad_w2**2] = [4,16]
      // w = [ w1_old - lr*grad_w1/sqrt(cahce_w2 + eps),
      //                w2_old - lr*grad_w1/sqrt(cahce_w2 + eps)]
      //                = [-0.1, -0.1]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw = session.activationArrayMap.get(w).getValues();
      test_util.expectArraysClose(dydw, new Float32Array([-.1, -0.1]), 1e-5);
      // cache = [old_cache_w1 + grad_w1**2,
      //                old_cache_w2 + grad_w2**2] = [4,16]
      // w = [ w1_old - lr*grad_w1/sqrt(cahce_w2 + eps),
      //                w2_old - lr*grad_w1/sqrt(cahce_w2 + eps)]
      //                = [-0.1707, -0.1707]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw2 = session.activationArrayMap.get(w).getValues();
      test_util.expectArraysClose(
          dydw2, new Float32Array([-.1707, -.1707]), 2e-5);
    });
  });

  it('Safe mode math, math scope train does not throw', () => {
    const x = g.placeholder('x', [2]);
    const w = g.variable('w', NDArray.zeros([1, 2]));
    const b = g.variable('b', NDArray.zeros([1]));
    const y = g.reduceSum(g.add(g.matmul(w, x), b));
    const safeMode = true;
    const optimizer = new RMSPropOptimizer(0.1, 0.8);
    const math = new NDArrayMathCPU(safeMode);
    const session = new Session(g, math);
    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([2, 4]);
      },
      disposeCopy(math, example) {}
    };

    math.scope(() => {
      // w = reduce_sum(w_1*x_1 + w_2*x_2 + b)
      // cache = [gamma*old_cache_w1 + (1-gamma)*grad_w1**2,
      //            gamma*old_cache_w2 + (1-gamma)*grad_w2**2]
      //            = [.8, .3.2]
      // w = [ w1_old - lr*grad_w1/sqrt(cahce_w1 + eps),
      //            w2_old - lr*grad_w1/sqrt(cahce_w2 + eps)]
      //            = [-0.2236, -0.2236]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw = session.activationArrayMap.get(w).getValues();
      test_util.expectArraysClose(
          dydw, new Float32Array([-.2236, -0.2236]), 1e-5);
      // cache = [gamma*old_cache_w1 + (1-gamma)*grad_w1**2,
      //            gamma*old_cache_w2 + (1-gamma)*grad_w2**2]
      //            = [1.44, 5.76]
      // w = [ w1_old - lr*grad_w1/sqrt(cahce_w1 + eps),
      //            w2_old - lr*grad_w1/sqrt(cahce_w2 + eps)]
      //            = [-.39027, -.39027]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw2 = session.activationArrayMap.get(w).getValues();
      test_util.expectArraysClose(
          dydw2, new Float32Array([-.39027, -.39027]), 2e-5);
    });
  });

  it('Safe mode math, no math scope train throws', () => {
    const x = g.placeholder('x', [2]);
    const y = g.square(x);
    const z = g.add(x, g.constant(3));
    const w = g.reduceSum(g.add(y, z));

    const safeMode = true;
    const optimizer = new SGDOptimizer(0.1);
    const math = new NDArrayMathCPU(safeMode);
    const session = new Session(g, math);
    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([2, 4]);
      },
      disposeCopy(math, example) {}
    };

    expect(
        () =>
            session.train(w, [{tensor: x, data: inputProvider}], 1, optimizer))
        .toThrowError();
  });
});
