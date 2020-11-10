/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import {getNoiseShape} from './dropout_util';

describeWithFlags('getNoiseShape', ALL_ENVS, () => {
  it('x.shape == noiseShape', async () => {
    const x = tf.ones([2, 3]);
    const noiseShape = [2, 3];
    const shape = getNoiseShape(x, noiseShape);
    expect(shape).toEqual([2, 3]);
  });

  it('x.shape and noiseShape have same length, different value', async () => {
    const x = tf.ones([2, 3]);
    const noiseShape = [2, 1];
    const shape = getNoiseShape(x, noiseShape);
    expect(shape).toEqual([2, 1]);
  });

  it('noiseShape has null value', async () => {
    const x = tf.ones([2, 3]);
    const noiseShape = [2, null];
    const shape = getNoiseShape(x, noiseShape);
    expect(shape).toEqual([2, 3]);
  });

  it('x.shape and noiseShape has different length', async () => {
    const x = tf.ones([2, 3, 4]);
    const noiseShape = [2, 3];
    const shape = getNoiseShape(x, noiseShape);
    expect(shape).toEqual([2, 3]);
  });

  it('noiseShape is null', async () => {
    const x = tf.ones([2, 3]);
    const shape = getNoiseShape(x, null);
    expect(shape).toEqual([2, 3]);
  });
});
