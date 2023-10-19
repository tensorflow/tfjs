/**
 * @license
 * Copyright 2023 Google LLC.
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
import {expectArraysClose} from '../test_util';
import {bitwiseAnd} from './bitwise_and';

describeWithFlags('bitwiseAnd', ALL_ENVS, () => {
  it('a bitwiseAnd b', async () => {
    const a = tf.tensor1d([0, 5, 3, 14], 'int32');
    const b = tf.tensor1d([5, 0, 7, 11], 'int32');

    const res = bitwiseAnd(a, b);
    expectArraysClose(await res.data(), [0, 0, 3, 10]);
  });

  it('different shape', () => {
    const a = tf.tensor1d([0, 5, 3, 14]);
    const b = tf.tensor1d([5, 0, 7]);

    expect(() => bitwiseAnd(a, b))
        .toThrowError(/BitwiseAnd: Tensors must have the same shape/);
  });

  it('wrong type', () => {
    const a = tf.tensor1d([0, 1, 3, 14], 'float32');
    const b = tf.tensor1d([5, 0, 7, 12], 'float32');

    expect(() => bitwiseAnd(a, b))
        .toThrowError(/BitwiseAnd: Only supports 'int32' values in tensor/);
  });
});
