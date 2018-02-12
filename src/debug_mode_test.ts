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

import * as dl from './index';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, describeWithFlags, expectArraysClose} from './test_util';
import * as util from './util';

const ALL_ENVS_DEBUG = ALL_ENVS.map(env => Object.assign({'DEBUG': true}, env));
const ALL_ENVS_NO_DEBUG =
    ALL_ENVS.map(env => Object.assign({'DEBUG': false}, env));

describeWithFlags('debug on', ALL_ENVS_DEBUG, () => {
  it('debug mode does not error when no nans', () => {
    const a = dl.tensor1d([2, -1, 0, 3]);
    const res = dl.relu(a);
    expectArraysClose(res, [2, 0, 0, 3]);
  });

  it('debug mode errors when there are nans, float32', () => {
    const a = dl.tensor1d([2, NaN]);
    const f = () => dl.relu(a);
    expect(f).toThrowError();
  });

  it('debug mode errors when there are nans, int32', () => {
    const a = dl.tensor1d([2, util.NAN_INT32], 'int32');
    const f = () => dl.relu(a);
    expect(f).toThrowError();
  });

  it('debug mode errors when there are nans, bool', () => {
    const a = dl.tensor1d([1, util.NAN_BOOL], 'bool');
    const f = () => dl.relu(a);
    expect(f).toThrowError();
  });

  it('A x B', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = dl.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = dl.matMul(a, b);

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(c, [0, 8, -3, 20]);
  });
});

describeWithFlags('debug off', ALL_ENVS_NO_DEBUG, () => {
  it('no errors where there are nans, and debug mode is disabled', () => {
    const a = dl.tensor1d([2, NaN]);
    const res = dl.relu(a);
    expectArraysClose(res, [2, NaN]);
  });
});
