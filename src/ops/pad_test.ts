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

import * as dl from '../index';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, describeWithFlags, expectArraysClose} from '../test_util';

describeWithFlags('pad1d', ALL_ENVS, () => {
  it('Should pad 1D arrays', () => {
    const a = dl.tensor1d([1, 2, 3, 4, 5, 6], 'int32');
    const b = dl.pad1d(a, [2, 3]);
    expectArraysClose(b, [0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0]);
  });

  it('Should not pad 1D arrays with 0s', () => {
    const a = dl.tensor1d([1, 2, 3, 4], 'int32');
    const b = dl.pad1d(a, [0, 0]);
    expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('Should handle padding with custom value', () => {
    let a = dl.tensor1d([1, 2, 3, 4], 'int32');
    let b = dl.pad1d(a, [2, 3], 9);
    expectArraysClose(b, [9, 9, 1, 2, 3, 4, 9, 9, 9]);

    a = dl.tensor1d([1, 2, 3, 4]);
    b = dl.pad1d(a, [2, 1], 1.1);
    expectArraysClose(b, [1.1, 1.1, 1, 2, 3, 4, 1.1]);

    a = dl.tensor1d([1, 2, 3, 4]);
    b = dl.pad1d(a, [2, 1], 1);
    expectArraysClose(b, [1, 1, 1, 2, 3, 4, 1]);
  });

  it('Should handle NaNs with 1D arrays', () => {
    const a = dl.tensor1d([1, NaN, 2, NaN]);
    const b = dl.pad1d(a, [1, 1]);
    expectArraysClose(b, [0, 1, NaN, 2, NaN, 0]);
  });

  it('Should handle invalid paddings', () => {
    const a = dl.tensor1d([1, 2, 3, 4], 'int32');
    const f = () => {
      // tslint:disable-next-line:no-any
      dl.pad1d(a, [2, 2, 2] as any);
    };
    expect(f).toThrowError();
  });

  it('grad', () => {
    const a = dl.tensor1d([1, 2, 3]);
    const dy = dl.tensor1d([10, 20, 30, 40, 50, 60]);
    const da = dl.grad((a: dl.Tensor1D) => dl.pad1d(a, [2, 1]))(a, dy);
    expect(da.shape).toEqual([3]);
    expectArraysClose(da, [30, 40, 50]);
  });
});

describeWithFlags('pad2d', ALL_ENVS, () => {
  it('Should pad 2D arrays', () => {
    let a = dl.tensor2d([[1], [2]], [2, 1], 'int32');
    let b = dl.pad2d(a, [[1, 1], [1, 1]]);
    // 0, 0, 0
    // 0, 1, 0
    // 0, 2, 0
    // 0, 0, 0
    expectArraysClose(b, [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0]);

    a = dl.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
    b = dl.pad2d(a, [[2, 2], [1, 1]]);
    // 0, 0, 0, 0, 0
    // 0, 0, 0, 0, 0
    // 0, 1, 2, 3, 0
    // 0, 4, 5, 6, 0
    // 0, 0, 0, 0, 0
    // 0, 0, 0, 0, 0
    expectArraysClose(b, [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0,
      0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
  });

  it('Should not pad 2D arrays with 0s', () => {
    const a = dl.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
    const b = dl.pad2d(a, [[0, 0], [0, 0]]);
    expectArraysClose(b, [1, 2, 3, 4, 5, 6]);
  });

  it('Should handle padding with custom value', () => {
    let a = dl.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
    let b = dl.pad2d(a, [[1, 1], [1, 1]], 10);
    expectArraysClose(b, [
      10, 10, 10, 10, 10, 10, 1,  2,  3,  10,
      10, 4,  5,  6,  10, 10, 10, 10, 10, 10
    ]);

    a = dl.tensor2d([[1], [1]], [2, 1]);
    b = dl.pad2d(a, [[1, 1], [1, 1]], -2.1);
    expectArraysClose(
        b, [-2.1, -2.1, -2.1, -2.1, 1, -2.1, -2.1, 1, -2.1, -2.1, -2.1, -2.1]);

    a = dl.tensor2d([[1], [1]], [2, 1]);
    b = dl.pad2d(a, [[1, 1], [1, 1]], -2);
    expectArraysClose(b, [-2, -2, -2, -2, 1, -2, -2, 1, -2, -2, -2, -2]);
  });

  it('Should handle NaNs with 2D arrays', () => {
    const a = dl.tensor2d([[1, NaN], [1, NaN]], [2, 2]);
    const b = dl.pad2d(a, [[1, 1], [1, 1]]);
    // 0, 0, 0,   0
    // 0, 1, NaN, 0
    // 0, 1, NaN, 0
    // 0, 0, 0,   0
    expectArraysClose(b, [0, 0, 0, 0, 0, 1, NaN, 0, 0, 1, NaN, 0, 0, 0, 0, 0]);
  });

  it('Should handle invalid paddings', () => {
    const a = dl.tensor2d([[1], [2]], [2, 1], 'int32');
    const f = () => {
      // tslint:disable-next-line:no-any
      dl.pad2d(a, [[2, 2, 2], [1, 1, 1]] as any);
    };
    expect(f).toThrowError();
  });

  it('grad', () => {
    const a = dl.tensor2d([[1, 2], [3, 4]]);
    const dy = dl.tensor2d([[0, 0, 0], [10, 20, 0], [30, 40, 0]], [3, 3]);
    const da =
        dl.grad((a: dl.Tensor2D) => dl.pad2d(a, [[1, 0], [0, 1]]))(a, dy);
    expect(da.shape).toEqual([2, 2]);
    expectArraysClose(da, [10, 20, 30, 40]);
  });
});

describeWithFlags('pad', ALL_ENVS, () => {
  it('Pad tensor2d', () => {
    let a = dl.tensor2d([[1], [2]], [2, 1], 'int32');
    let b = dl.pad(a, [[1, 1], [1, 1]]);
    // 0, 0, 0
    // 0, 1, 0
    // 0, 2, 0
    // 0, 0, 0
    expectArraysClose(b, [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0]);

    a = dl.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
    b = dl.pad(a, [[2, 2], [1, 1]]);
    // 0, 0, 0, 0, 0
    // 0, 0, 0, 0, 0
    // 0, 1, 2, 3, 0
    // 0, 4, 5, 6, 0
    // 0, 0, 0, 0, 0
    // 0, 0, 0, 0, 0
    expectArraysClose(b, [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0,
      0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
  });
});
