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
import {Array1D, Array2D, NDArray} from './ndarray';

const tests: MathTests = it => {
  const NUM_SAMPLES = 10000;
  // Allowed Variance in probability (in %).
  const EPSILON = 0.05;

  it('Flip a fair coin and check bounds', math => {
    const probs = Array1D.new([0.5, 0.5]);
    const result = math.multinomial(probs, NUM_SAMPLES);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([NUM_SAMPLES]);
    const outcomeProbs = computeProbs(result.dataSync(), 2);
    test_util.expectArraysClose(outcomeProbs, [0.5, 0.5], EPSILON);
  });

  it('Flip a two-sided coin with 100% of heads', math => {
    const probs = Array1D.new([1, 0]);
    const result = math.multinomial(probs, NUM_SAMPLES);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([NUM_SAMPLES]);
    const outcomeProbs = computeProbs(result.dataSync(), 2);
    test_util.expectArraysClose(outcomeProbs, [1, 0], EPSILON);
  });

  it('Flip a two-sided coin with 100% of tails', math => {
    const probs = Array1D.new([0, 1]);
    const result = math.multinomial(probs, NUM_SAMPLES);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([NUM_SAMPLES]);
    const outcomeProbs = computeProbs(result.dataSync(), 2);
    test_util.expectArraysClose(outcomeProbs, [0, 1], EPSILON);
  });

  it('Flip a single-sided coin throws error', math => {
    const probs = Array1D.new([1]);
    expect(() => math.multinomial(probs, NUM_SAMPLES)).toThrowError();
  });

  it('Flip a ten-sided coin and check bounds', math => {
    const numOutcomes = 10;
    const probs = Array1D.zeros([numOutcomes]);
    for (let i = 0; i < numOutcomes; ++i) {
      probs.set(1 / numOutcomes, i);
    }
    const result = math.multinomial(probs, NUM_SAMPLES);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([NUM_SAMPLES]);
    const outcomeProbs = computeProbs(result.dataSync(), numOutcomes);
    expect(outcomeProbs.length).toBeLessThanOrEqual(numOutcomes);
  });

  it('Flip 3 three-sided coins, each coin is 100% biases', math => {
    const numOutcomes = 3;
    const probs =
        Array2D.new([3, numOutcomes], [[0, 0, 1], [0, 1, 0], [1, 0, 0]]);
    const result = math.multinomial(probs, NUM_SAMPLES);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([3, NUM_SAMPLES]);

    // First coin always gets last event.
    let outcomeProbs =
        computeProbs(result.dataSync().slice(0, NUM_SAMPLES), numOutcomes);
    test_util.expectArraysClose(outcomeProbs, [0, 0, 1], EPSILON);

    // Second coin always gets middle event.
    outcomeProbs = computeProbs(
        result.dataSync().slice(NUM_SAMPLES, 2 * NUM_SAMPLES), numOutcomes);
    test_util.expectArraysClose(outcomeProbs, [0, 1, 0], EPSILON);

    // Third coin always gets first event
    outcomeProbs =
        computeProbs(result.dataSync().slice(2 * NUM_SAMPLES), numOutcomes);
    test_util.expectArraysClose(outcomeProbs, [1, 0, 0], EPSILON);

  });

  it('passing Array3D throws error', math => {
    const probs = NDArray.zeros([3, 2, 2]) as Array1D;
    expect(() => math.multinomial(probs, 3)).toThrowError();
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
};

test_util.describeMathCPU('multinomial', [tests]);
test_util.describeMathGPU('multinomial', [tests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
