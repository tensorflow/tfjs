/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as dl from 'deeplearn';
// tslint:disable-next-line:max-line-length
import {expectArraysClose, expectArraysEqual, expectNumbersClose} from 'deeplearn/dist/test_util';

import {bindTensorFlowBackend} from '.';

// BeforeEach?
bindTensorFlowBackend();

describe('matMul', () => {
  it('should work', () => {
    const t1 = dl.tensor2d([[1, 2], [3, 4]]);
    const t2 = dl.tensor2d([[5, 6], [7, 8]]);
    const result = t1.matMul(t2);
    expectArraysClose(result, [19, 22, 43, 50]);
  });
});

describe('slice tensor1d', () => {
  it('slices 1x1 into 1x1 (effectively a copy)', () => {
    const a = dl.tensor1d([5]);
    const result = dl.slice1d(a, 0, 1);

    expect(result.shape).toEqual([1]);
    expect(result.get(0)).toEqual(5);
  });

  it('slices 5x1 into shape 2x1 starting at 3', () => {
    const a = dl.tensor1d([1, 2, 3, 4, 5]);
    const result = dl.slice1d(a, 3, 2);

    expect(result.shape).toEqual([2]);
    expectArraysClose(result, [4, 5]);
  });

  it('slices 5x1 into shape 3x1 starting at 1', () => {
    const a = dl.tensor1d([1, 2, 3, 4, 5]);
    const result = dl.slice1d(a, 1, 3);

    expect(result.shape).toEqual([3]);
    expectArraysClose(result, [2, 3, 4]);
  });
});

describe('reshape', () => {
  it('should work', () => {
    const a = dl.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
  });
});

describe('cast', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, 2, 3, 4, 5], 'int32');
    const b = dl.cast(a, 'float32');
    expect(b.dtype).toBe('float32');
  });
});

describe('pad', () => {
  it('should work', () => {
    const t = dl.tensor2d([[1, 1], [1, 1]]);
    const result = dl.pad2d(t, [[1, 1], [1, 1]]);
    expectArraysClose(result, [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]);
  });
});

describe('reverse', () => {
  it('should work', () => {
    const input = dl.tensor1d([1, 2, 3, 4, 5]);
    const result = dl.reverse(input);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(result, [5, 4, 3, 2, 1]);
  });
});

describe('concat', () => {
  it('should work', () => {
    const a = dl.tensor1d([3]);
    const b = dl.tensor1d([5]);

    const result = dl.concat1d([a, b]);
    const expected = [3, 5];
    expectArraysClose(result, expected);
  });

  it('should work with 2darray', () => {
    const a = dl.ones([1, 10], 'int32');
    const b = dl.ones([1, 10], 'int32');
    const c = dl.concat([a, b]);
    expect(c.shape).toEqual([2, 10]);
    const d = dl.concat([c, a]);
    expect(d.shape).toEqual([3, 10]);
  });
});

describe('neg', () => {
  it('should work', () => {
    const input = dl.tensor1d([1, 2, 3, 4, 5]);
    const result = dl.neg(input);
    expectArraysClose(result, [-1, -2, -3, -4, -5]);
  });
});

describe('add', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, 1]);
    const b = dl.tensor1d([2, 2]);
    expectArraysClose(a.add(b), [3, 3]);
  });
});

describe('sub', () => {
  it('should work', () => {
    const a = dl.tensor1d([2, 2]);
    const b = dl.tensor1d([1, 1]);
    expectArraysClose(a.sub(b), [1, 1]);
  });
});

describe('multiply', () => {
  it('should work', () => {
    const a = dl.tensor1d([2, 2]);
    const b = dl.tensor1d([2, 2]);
    expectArraysClose(a.mul(b), [4, 4]);
  });
});

describe('div', () => {
  it('should work', () => {
    const a = dl.tensor1d([4, 4]);
    const b = dl.tensor1d([2, 2]);
    expectArraysClose(a.div(b), [2, 2]);
  });
});

describe('sum', () => {
  it('should work', () => {
    const a = dl.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const result = dl.sum(a);
    expect(result.get()).toEqual(7);
  });
});

describe('equal', () => {
  it('should work', () => {
    const a = dl.tensor1d([4, 2]);
    const b = dl.tensor1d([2, 2]);
    expectArraysClose(a.equal(b), [0, 1]);
  });
});

describe('notEqual', () => {
  it('should work', () => {
    const a = dl.tensor1d([4, 2]);
    const b = dl.tensor1d([2, 2]);
    expectArraysClose(a.notEqual(b), [1, 0]);
  });
});

describe('less', () => {
  it('should work', () => {
    const a = dl.tensor1d([4, 1]);
    const b = dl.tensor1d([2, 2]);
    expectArraysClose(a.less(b), [0, 1]);
  });
});

describe('lessEqual', () => {
  it('should work', () => {
    const a = dl.tensor1d([4, 1, 3]);
    const b = dl.tensor1d([2, 2, 3]);
    expectArraysClose(a.lessEqual(b), [0, 1, 1]);
  });
});

describe('greater', () => {
  it('should work', () => {
    const a = dl.tensor1d([4, 1]);
    const b = dl.tensor1d([2, 2]);
    expectArraysClose(a.greater(b), [1, 0]);
  });
});

describe('greaterEqual', () => {
  it('should work', () => {
    const a = dl.tensor1d([4, 1, 3]);
    const b = dl.tensor1d([2, 2, 3]);
    expectArraysClose(a.greaterEqual(b), [1, 0, 1]);
  });
});

describe('logicalNot', () => {
  it('should work', () => {
    const a = dl.tensor1d([0, 1, 1], 'bool');
    expectArraysClose(dl.logicalNot(a), [1, 0, 0]);
  });
});

describe('logicalAnd', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, 0, 1], 'bool');
    const b = dl.tensor1d([0, 1, 1], 'bool');
    expectArraysClose(a.logicalAnd(b), [0, 0, 1]);
  });
});

describe('logicalOr', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, 0, 0], 'bool');
    const b = dl.tensor1d([0, 1, 0], 'bool');
    expectArraysClose(a.logicalOr(b), [1, 1, 0]);
  });
});

describe('min', () => {
  it('should work', () => {
    const a = dl.tensor1d([3, -1, 0, 100, -7, 2]);
    expectNumbersClose(dl.min(a).get(), -7);
  });
});

describe('minimum', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, 5, 2, 3], 'int32');
    const b = dl.tensor1d([2, 3, 1, 4], 'int32');
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [1, 3, 1, 3]);
  });
});

describe('max', () => {
  it('should work', () => {
    const a = dl.tensor1d([3, -1, 0, 100, -7, 2]);
    expectNumbersClose(dl.max(a).get(), 100);
  });
});

describe('maximum', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, 5, 2, 3], 'int32');
    const b = dl.tensor1d([2, 3, 1, 4], 'int32');
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [2, 5, 2, 4]);
  });
});

describe('ceil', () => {
  it('should work', () => {
    const a = dl.tensor1d([1.5, 2.1, -1.4]);
    const r = dl.ceil(a);
    expectNumbersClose(r.get(0), 2);
    expectNumbersClose(r.get(1), 3);
    expectNumbersClose(r.get(2), -1);
  });
});

describe('floor', () => {
  it('should work', () => {
    const a = dl.tensor1d([1.5, 2.1, -1.4]);
    const r = dl.floor(a);
    expectNumbersClose(r.get(0), 1);
    expectNumbersClose(r.get(1), 2);
    expectNumbersClose(r.get(2), -2);
  });
});

describe('pow', () => {
  it('should work', () => {
    const a = dl.tensor2d([1, -2, -3, 0, 7, 1], [2, 3]);
    const b = dl.tensor2d([5, 3, 4, 5, 2, -3], [2, 3]);
    const expected = [1, -8, 81, 0, 49, 1];
    const result = dl.pow(a, b);

    expect(result.shape).toEqual([2, 3]);
    expectArraysClose(result, expected, 0.01);
  });
});

describe('exp', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, 2, 0]);
    const r = dl.exp(a);

    expectNumbersClose(r.get(0), Math.exp(1));
    expectNumbersClose(r.get(1), Math.exp(2));
    expectNumbersClose(r.get(2), 1);
  });
});

describe('log', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, 2]);
    const r = dl.log(a);
    expectNumbersClose(r.get(0), Math.log(1));
    expectNumbersClose(r.get(1), Math.log(2));
  });
});

describe('sqrt', () => {
  it('should work', () => {
    const a = dl.tensor1d([2, 4]);
    const r = dl.sqrt(a);
    expectNumbersClose(r.get(0), Math.sqrt(2));
    expectNumbersClose(r.get(1), Math.sqrt(4));
  });
});

describe('square', () => {
  it('should work', () => {
    const a = dl.tensor1d([2, 4, Math.sqrt(2)]);
    const r = dl.square(a);
    expectArraysClose(r, [4, 16, 2]);
  });
});

describe('relu', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, -2, 0, 3, -0.1]);
    expectArraysClose(dl.relu(a), [1, 0, 0, 3, 0]);
  });
});

describe('elu', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, -1, 0]);
    const result = dl.elu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1, -0.6321, 0]);
  });
});

describe('selu', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, -1, 0]);
    const result = dl.selu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1.0507, -1.1113, 0]);
  });
});

describe('abs', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, -2, 0, 3, -0.1]);
    const result = dl.abs(a);
    expectArraysClose(result, [1, 2, 0, 3, 0.1]);
  });
});

describe('sigmoid', () => {
  it('should work', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);

    const result = dl.sigmoid(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = 1 / (1 + Math.exp(-values[i]));
    }
    expectArraysClose(result, expected);
  });
});

describe('sin', () => {
  it('should work', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.sin(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.sin(values[i]);
    }
    expectArraysClose(result, expected);
  });
});

describe('cos', () => {
  it('should work', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.cos(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.cos(values[i]);
    }
    expectArraysClose(result, expected);
  });
});

describe('tan', () => {
  it('should work', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.tan(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.tan(values[i]);
    }
    expectArraysClose(result, expected);
  });
});

describe('asin', () => {
  it('should work', () => {
    const values = [.1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.asin(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.asin(values[i]);
    }
    expectArraysClose(result, expected);
  });
});

describe('acos', () => {
  it('should work', () => {
    const values = [.1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.acos(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.acos(values[i]);
    }
    expectArraysClose(result, expected);
  });
});

describe('atan', () => {
  it('should work', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.atan(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atan(values[i]);
    }
    expectArraysClose(result, expected);
  });
});

describe('sinh', () => {
  it('should work', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.sinh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.sinh(values[i]);
    }
    expectArraysClose(result, expected);
  });
});

describe('cosh', () => {
  it('should work', () => {
    const values = [1, -3, 2, -1, -4];
    const a = dl.tensor1d(values);
    const result = dl.cosh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.cosh(values[i]);
    }

    expectArraysClose(result, expected);
  });
});

describe('tanh', () => {
  it('should work', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.tanh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.tanh(values[i]);
    }
    expectArraysClose(result, expected);
  });
});

describe('oneHot', () => {
  it('should work', () => {
    const indices = dl.tensor1d([0, 1], 'int32');
    const res = dl.oneHot(indices, 2);

    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(res, [1, 0, 0, 1]);
  });
});
