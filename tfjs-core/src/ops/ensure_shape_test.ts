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
import {ensureShape} from './ensure_shape';

describeWithFlags('ensureShape', ALL_ENVS, () => {
  it('basic', () => {
    const x = tf.ones([2, 3]);
    expect(ensureShape(x, [2, 3])).toEqual(x);
  });

  it('different shape', () => {
    const x = tf.ones([2, 3]);
    expect(() => ensureShape(x, [5, 3])).toThrowError(/EnsureShape:/);
  });

  it('different length', () => {
    const x = tf.tensor1d([1, 2, 3, 4]);
    expect(() => ensureShape(x, [1, 3])).toThrowError(/EnsureShape:/);
  });

  it('null shape', () => {
    const x = tf.tensor2d([1, null, 3, 4], [2, 2]);
    expect(ensureShape(x, [2, 2])).toEqual(x);
  });
});
