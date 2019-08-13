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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {Tensor1D} from '../tensor';
import {expectArraysClose} from '../test_util';

describeWithFlags('multinomial', ALL_ENVS, () => {
  const NUM_SAMPLES = 1000;
  // Allowed Variance in probability (in %).
  const EPSILON = 0.05;
  const SEED = 3.14;

  it('Flip a fair coin and check bounds', async () => {
    const probs = tf.tensor1d([1, 1]);
    const result = tf.multinomial(probs, NUM_SAMPLES, SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([NUM_SAMPLES]);
    const outcomeProbs = computeProbs(await result.data(), 2);
    expectArraysClose(outcomeProbs, [0.5, 0.5], EPSILON);
  });

  it('Flip a two-sided coin with 100% of heads', async () => {
    const logits = tf.tensor1d([1, -100]);
    const result = tf.multinomial(logits, NUM_SAMPLES, SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([NUM_SAMPLES]);
    const outcomeProbs = computeProbs(await result.data(), 2);
    expectArraysClose(outcomeProbs, [1, 0], EPSILON);
  });

  it('Flip a two-sided coin with 100% of tails', async () => {
    const logits = tf.tensor1d([-100, 1]);
    const result = tf.multinomial(logits, NUM_SAMPLES, SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([NUM_SAMPLES]);
    const outcomeProbs = computeProbs(await result.data(), 2);
    expectArraysClose(outcomeProbs, [0, 1], EPSILON);
  });

  it('Flip a single-sided coin throws error', () => {
    const probs = tf.tensor1d([1]);
    expect(() => tf.multinomial(probs, NUM_SAMPLES, SEED)).toThrowError();
  });

  it('Flip a ten-sided coin and check bounds', async () => {
    const numOutcomes = 10;
    const logits = tf.fill([numOutcomes], 1).as1D();
    const result = tf.multinomial(logits, NUM_SAMPLES, SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([NUM_SAMPLES]);
    const outcomeProbs = computeProbs(await result.data(), numOutcomes);
    expect(outcomeProbs.length).toBeLessThanOrEqual(numOutcomes);
  });

  it('Flip 3 three-sided coins, each coin is 100% biases', async () => {
    const numOutcomes = 3;
    const logits = tf.tensor2d(
        [[-100, -100, 1], [-100, 1, -100], [1, -100, -100]], [3, numOutcomes]);
    const result = tf.multinomial(logits, NUM_SAMPLES, SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([3, NUM_SAMPLES]);

    // First coin always gets last event.
    let outcomeProbs =
        computeProbs((await result.data()).slice(0, NUM_SAMPLES), numOutcomes);
    expectArraysClose(outcomeProbs, [0, 0, 1], EPSILON);

    // Second coin always gets middle event.
    outcomeProbs = computeProbs(
        (await result.data()).slice(NUM_SAMPLES, 2 * NUM_SAMPLES), numOutcomes);
    expectArraysClose(outcomeProbs, [0, 1, 0], EPSILON);

    // Third coin always gets first event
    outcomeProbs =
        computeProbs((await result.data()).slice(2 * NUM_SAMPLES), numOutcomes);
    expectArraysClose(outcomeProbs, [1, 0, 0], EPSILON);
  });

  it('passing Tensor3D throws error', () => {
    const probs = tf.zeros([3, 2, 2]);
    const normalized = true;
    expect(() => tf.multinomial(probs as Tensor1D, 3, SEED, normalized))
        .toThrowError();
  });

  it('throws when passed a non-tensor', () => {
    // tslint:disable-next-line:no-any
    expect(() => tf.multinomial({} as any, NUM_SAMPLES, SEED))
        .toThrowError(
            /Argument 'logits' passed to 'multinomial' must be a Tensor/);
  });

  it('accepts a tensor-like object for logits (biased coin)', async () => {
    const res = tf.multinomial([-100, 1], NUM_SAMPLES, SEED);
    expect(res.dtype).toBe('int32');
    expect(res.shape).toEqual([NUM_SAMPLES]);
    const outcomeProbs = computeProbs(await res.data(), 2);
    expectArraysClose(outcomeProbs, [0, 1], EPSILON);
  });

  function computeProbs(
      events: Float32Array|Uint8Array|Int32Array, numOutcomes: number) {
    const counts = [];
    for (let i = 0; i < numOutcomes; ++i) {
      counts[i] = 0;
    }
    const numSamples = events.length;
    for (let i = 0; i < events.length; ++i) {
      counts[events[i]]++;
    }
    // Normalize counts to be probabilities between [0, 1].
    for (let i = 0; i < counts.length; i++) {
      counts[i] /= numSamples;
    }
    return counts;
  }
});
