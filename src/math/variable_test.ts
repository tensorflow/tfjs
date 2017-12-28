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

import * as test_util from '../test_util';
import {MathTests} from '../test_util';
import {Array1D, Array2D} from './ndarray';
import {variable, Variable} from './variable';

const tests: MathTests = it => {
  it('simple update', math => {
    const v = variable(Array1D.new([1, 2, 3]));
    test_util.expectArraysClose(v, [1, 2, 3]);

    v.assign(Array1D.new([4, 5, 6]));
    test_util.expectArraysClose(v, [4, 5, 6]);
  });

  it('default names are unique', math => {
    const v = variable(Array1D.new([1, 2, 3]));
    expect(v.name).not.toBeNull();

    const v2 = variable(Array1D.new([1, 2, 3]));
    expect(v2.name).not.toBeNull();
    expect(v.name).not.toBe(v2.name);
  });

  it('user provided name', math => {
    const v = variable(Array1D.new([1, 2, 3]), true, 'myName');
    expect(v.name).toBe('myName');
  });

  it('if name already used, throw error', math => {
    variable(Array1D.new([1, 2, 3]), true, 'myName');
    expect(() => variable(Array1D.new([1, 2, 3]), true, 'myName'))
        .toThrowError();
  });

  it('math ops can take variables', math => {
    const value = Array1D.new([1, 2, 3]);
    const v = variable(value);
    const res = math.sum(v);
    test_util.expectArraysClose(res, [6]);
  });

  it('variables are not affected by scopes', math => {
    let v: Variable<'float32', '1'>;
    expect(math.getNumArrays()).toBe(0);

    math.scope(() => {
      const value = Array1D.new([1, 2, 3], 'float32');
      expect(math.getNumArrays()).toBe(1);

      v = variable(value);
      expect(math.getNumArrays()).toBe(1);
    });

    expect(math.getNumArrays()).toBe(1);
    test_util.expectArraysClose(v, [1, 2, 3]);

    v.dispose();
    expect(math.getNumArrays()).toBe(0);
  });

  it('update will dispose old data', math => {
    let v: Variable<'float32', '1'>;
    const firstValue = Array1D.new([1, 2, 3]);

    v = variable(firstValue);
    expect(math.getNumArrays()).toBe(1);

    const secondValue = Array1D.new([4, 5, 6]);
    expect(math.getNumArrays()).toBe(2);

    v.assign(secondValue);
    expect(math.getNumArrays()).toBe(1);
    // The first value was disposed.
    expect(() => firstValue.dataSync()).toThrowError();

    v.dispose();
    expect(math.getNumArrays()).toBe(0);
    // The second value was disposed.
    expect(() => secondValue.dataSync()).toThrowError();
  });

  it('shape must match', math => {
    const v = variable(Array1D.new([1, 2, 3]));
    expect(() => v.assign(Array1D.new([1, 2]))).toThrowError();
    // tslint:disable-next-line:no-any
    expect(() => v.assign(Array2D.new([1, 2], [3, 4]) as any)).toThrowError();
  });

  it('dtype must match', math => {
    const v = variable(Array1D.new([1, 2, 3]));
    // tslint:disable-next-line:no-any
    expect(() => v.assign(Array1D.new([1, 1, 1], 'int32') as any))
        .toThrowError();
    // tslint:disable-next-line:no-any
    expect(() => v.assign(Array1D.new([true, false, true], 'bool') as any))
        .toThrowError();
  });
};

test_util.describeMathCPU('Variables', [tests]);
test_util.describeMathGPU('Variables', [tests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
