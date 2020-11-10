/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

describeWithFlags('complex64', ALL_ENVS, () => {
  it('tf.complex', async () => {
    const real = tf.tensor1d([3, 30]);
    const imag = tf.tensor1d([4, 40]);
    const complex = tf.complex(real, imag);

    expect(complex.dtype).toBe('complex64');
    expect(complex.shape).toEqual(real.shape);
    expectArraysClose(await complex.data(), [3, 4, 30, 40]);
  });

  it('tf.real', async () => {
    const complex = tf.complex([3, 30], [4, 40]);
    const real = tf.real(complex);

    expect(real.dtype).toBe('float32');
    expect(real.shape).toEqual([2]);
    expectArraysClose(await real.data(), [3, 30]);
  });

  it('tf.imag', async () => {
    const complex = tf.complex([3, 30], [4, 40]);
    const imag = tf.imag(complex);

    expect(imag.dtype).toBe('float32');
    expect(imag.shape).toEqual([2]);
    expectArraysClose(await imag.data(), [4, 40]);
  });

  it('throws when shapes dont match', () => {
    const real = tf.tensor1d([3, 30]);
    const imag = tf.tensor1d([4, 40, 50]);

    const re =
        /real and imag shapes, 2 and 3, must match in call to tf.complex\(\)/;
    expect(() => tf.complex(real, imag)).toThrowError(re);
  });
});
