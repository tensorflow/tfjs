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

import * as tf from '@tensorflow/tfjs';

import {arrayStats, tensorStats} from './math';

//
// arrayStats
//
describe('arrayStats', () => {
  it('throws on non array input', () => {
    // @ts-ignore
    expect(() => arrayStats('string')).toThrow();
  });

  it('handles empty arrays', () => {
    const stats = arrayStats([]);
    expect(stats.max).toBe(undefined);
    expect(stats.min).toBe(undefined);
    expect(stats.numVals).toBe(0);
    expect(stats.numNans).toBe(0);
    expect(stats.numZeros).toBe(0);
  });

  it('computes correct stats', () => {
    const data = [2, 3, -400, 500, NaN, -800, 0, 0, 0];
    const stats = arrayStats(data);

    expect(stats.max).toBe(500);
    expect(stats.min).toBe(-800);
    expect(stats.numVals).toBe(9);
    expect(stats.numNans).toBe(1);
    expect(stats.numZeros).toBe(3);
  });

  it('computes correct stats — all negative', () => {
    const data = [-2, -3, -400, -500, NaN, -800];
    const stats = arrayStats(data);
    expect(stats.max).toBe(-2);
    expect(stats.min).toBe(-800);
    expect(stats.numVals).toBe(6);
    expect(stats.numNans).toBe(1);
    expect(stats.numZeros).toBe(0);
  });

  it('computes correct stats — all zeros', () => {
    const data = [0, 0, 0, 0];
    const stats = arrayStats(data);
    expect(stats.max).toBe(0);
    expect(stats.min).toBe(0);
    expect(stats.numVals).toBe(4);
    expect(stats.numNans).toBe(0);
    expect(stats.numZeros).toBe(4);
  });

  it('computes correct stats — all NaNs', () => {
    const data = [NaN, NaN, NaN, NaN];
    const stats = arrayStats(data);
    expect(isNaN(stats.max!)).toBe(true);
    expect(isNaN(stats.min!)).toBe(true);
    expect(stats.numVals).toBe(4);
    expect(stats.numNans).toBe(4);
    expect(stats.numZeros).toBe(0);
  });
});

//
// tensorStats
//
describe('tensorStats', () => {
  it('computes correct stats', async () => {
    const data = tf.tensor([2, 3, -400, 500, NaN, -800, 0, 0, 0]);
    const stats = await tensorStats(data);

    expect(stats.max).toBe(500);
    expect(stats.min).toBe(-800);
    expect(stats.numVals).toBe(9);
    expect(stats.numNans).toBe(1);
    expect(stats.numZeros).toBe(3);
  });

  it('computes correct stats — all negative', async () => {
    const data = tf.tensor([-2, -3, -400, -500, NaN, -800]);
    const stats = await tensorStats(data);
    expect(stats.max).toBe(-2);
    expect(stats.min).toBe(-800);
    expect(stats.numVals).toBe(6);
    expect(stats.numNans).toBe(1);
    expect(stats.numZeros).toBe(0);
  });

  it('computes correct stats — all zeros', async () => {
    const data = tf.tensor([0, 0, 0, 0]);
    const stats = await tensorStats(data);
    expect(stats.max).toBe(0);
    expect(stats.min).toBe(0);
    expect(stats.numVals).toBe(4);
    expect(stats.numNans).toBe(0);
    expect(stats.numZeros).toBe(4);
  });

  it('computes correct stats — all NaNs', async () => {
    const data = tf.tensor([NaN, NaN, NaN, NaN]);
    const stats = await tensorStats(data);
    expect(isNaN(stats.max!)).toBe(true);
    expect(isNaN(stats.min!)).toBe(true);
    expect(stats.numVals).toBe(4);
    expect(stats.numNans).toBe(4);
    expect(stats.numZeros).toBe(0);
  });
});
