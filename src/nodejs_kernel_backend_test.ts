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
import {expectArraysClose} from 'deeplearn/dist/test_util';

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

describe('pad', () => {
  it('should work', () => {
    const t = dl.tensor2d([[1, 1], [1, 1]]);
    const result = dl.pad2d(t, [[1, 1], [1, 1]]);
    expectArraysClose(result, [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]);
  });
});

describe('relu', () => {
  it('should work', () => {
    const a = dl.tensor1d([1, -2, 0, 3, -0.1]);
    expectArraysClose(dl.relu(a), [1, 0, 0, 3, 0]);
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
