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

import * as dl from './index';
import {ALL_ENVS, describeWithFlags, expectArraysClose} from './test_util';
// tslint:disable-next-line:max-line-length
import {Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, variable, Variable} from './tensor';
import {Rank} from './types';

describeWithFlags('variable', ALL_ENVS, () => {
  it('simple assign', () => {
    const v = variable(dl.tensor1d([1, 2, 3]));
    expectArraysClose(v, [1, 2, 3]);

    v.assign(dl.tensor1d([4, 5, 6]));
    expectArraysClose(v, [4, 5, 6]);
  });

  it('default names are unique', () => {
    const v = variable(dl.tensor1d([1, 2, 3]));
    expect(v.name).not.toBeNull();

    const v2 = variable(dl.tensor1d([1, 2, 3]));
    expect(v2.name).not.toBeNull();
    expect(v.name).not.toBe(v2.name);
  });

  it('user provided name', () => {
    const v = variable(dl.tensor1d([1, 2, 3]), true, 'myName');
    expect(v.name).toBe('myName');
  });

  it('if name already used, throw error', () => {
    variable(dl.tensor1d([1, 2, 3]), true, 'myName');
    expect(() => variable(dl.tensor1d([1, 2, 3]), true, 'myName'))
        .toThrowError();
  });

  it('math ops can take variables', () => {
    const value = dl.tensor1d([1, 2, 3]);
    const v = variable(value);
    const res = dl.sum(v);
    expectArraysClose(res, [6]);
  });

  it('variables are not affected by tidy', () => {
    let v: Variable<Rank.R1>;
    expect(dl.memory().numTensors).toBe(0);

    dl.tidy(() => {
      const value = dl.tensor1d([1, 2, 3], 'float32');
      expect(dl.memory().numTensors).toBe(1);

      v = variable(value);
      expect(dl.memory().numTensors).toBe(1);
    });

    expect(dl.memory().numTensors).toBe(1);
    expectArraysClose(v, [1, 2, 3]);

    v.dispose();
    expect(dl.memory().numTensors).toBe(0);
  });

  it('variables are assignable to tensors', () => {
    // This test asserts compilation, not doing any run-time assertion.
    const x0: Variable<Rank.R0> = null;
    const y0: Scalar = x0;
    expect(y0).toBeNull();

    const x1: Variable<Rank.R1> = null;
    const y1: Tensor1D = x1;
    expect(y1).toBeNull();

    const x2: Variable<Rank.R2> = null;
    const y2: Tensor2D = x2;
    expect(y2).toBeNull();

    const x3: Variable<Rank.R3> = null;
    const y3: Tensor3D = x3;
    expect(y3).toBeNull();

    const x4: Variable<Rank.R4> = null;
    const y4: Tensor4D = x4;
    expect(y4).toBeNull();

    const xh: Variable = null;
    const yh: Tensor = xh;
    expect(yh).toBeNull();
  });

  it('assign will dispose old data', () => {
    let v: Variable<Rank.R1>;
    v = variable(dl.tensor1d([1, 2, 3]));
    expect(dl.memory().numTensors).toBe(1);
    expectArraysClose(v, [1, 2, 3]);

    const secondArray = dl.tensor1d([4, 5, 6]);
    expect(dl.memory().numTensors).toBe(2);

    v.assign(secondArray);
    expectArraysClose(v, [4, 5, 6]);
    // Assign doesn't dispose the 1st array.
    expect(dl.memory().numTensors).toBe(2);

    v.dispose();
    // Disposing the variable disposes itself.
    expect(dl.memory().numTensors).toBe(1);
  });

  it('shape must match', () => {
    const v = variable(dl.tensor1d([1, 2, 3]));
    expect(() => v.assign(dl.tensor1d([1, 2]))).toThrowError();
    // tslint:disable-next-line:no-any
    expect(() => v.assign(dl.tensor2d([3, 4], [1, 2]) as any)).toThrowError();
  });

  it('dtype must match', () => {
    const v = variable(dl.tensor1d([1, 2, 3]));
    // tslint:disable-next-line:no-any
    expect(() => v.assign(dl.tensor1d([1, 1, 1], 'int32') as any))
        .toThrowError();
    // tslint:disable-next-line:no-any
    expect(() => v.assign(dl.tensor1d([true, false, true], 'bool') as any))
        .toThrowError();
  });
});
