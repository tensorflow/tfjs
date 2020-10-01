/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysEqual} from '../test_util';

import {tensor1d} from './tensor1d';

describeWithFlags('unique', ALL_ENVS, () => {
  it('1d tensor with int32', async () => {
    const x = tensor1d([1, 1, 2, 4, 4, 4, 7, 8, 8]);
    const {values, indices} = tf.unique(x);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual(x.shape);
    expectArraysEqual(await values.data(), [1, 2, 4, 7, 8]);
    expectArraysEqual(await indices.data(), [0, 0, 1, 2, 2, 2, 3, 4, 4]);
  });

  it('1d tensor with string', async () => {
    const x = tensor1d(['a', 'b', 'b', 'c', 'c']);
    const {values, indices} = tf.unique(x);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual(x.shape);
    expectArraysEqual(await values.data(), ['a', 'b', 'c']);
    expectArraysEqual(await indices.data(), [0, 1, 1, 2, 2]);
  });

  it('1d tensor with bool', async () => {
    const x = tensor1d([true, true, false]);
    const {values, indices} = tf.unique(x);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual(x.shape);
    expectArraysEqual(await values.data(), [true, false]);
    expectArraysEqual(await indices.data(), [0, 0, 1]);
  });

  it('throws for non 1-D tensor', () => {
    expect(() => tf.unique([[1, 2], [3, 4]]))
        .toThrowError(
            /unique\(\) currently only supports 1-D tensor.*got rank 2.*/);
  });
});
