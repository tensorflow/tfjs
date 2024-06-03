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

/**
 * Unit Tests for Sampler Layers.
 */

import { Tensor, fill, oneHot, ones, range, tensor, test_util, zeros } from '@tensorflow/tfjs-core';

import { NextFn, TopKSampler } from './samplers';
import { describeMathCPU } from '../../utils/test_utils';

describeMathCPU('TopKSampler', () => {
  const intLookup: {[key: number]: string} = {};
  const charLookup: {[key: string]: number} = {};
  let batchSize: number;
  let length: number;
  let vocabSize: number;
  let next: NextFn;
  let sampler: TopKSampler;

  const joinAsString = (x: Tensor) => {
    const tensorArr = x.arraySync() as number[][];
    return tensorArr.map((numArr) => numArr.map(i => intLookup[i]).join(''));
  };

  beforeAll(() => {
    const charCodes = Array(26);
    for (const charCode of charCodes.keys()) {
      const char = String.fromCharCode(charCode + 97);
      intLookup[charCode] = char;
      charLookup[char] = charCode;
    }

    batchSize = 1;
    length = 12;
    vocabSize = Object.keys(intLookup).length;

    function nextFn(
      prompt: Tensor, cache: Tensor, index: number
    ): [Tensor, Tensor, Tensor] {
      // Dummy hidden states.
      const hiddenStates = ones([batchSize, 5]);
      // Return a distribution favoring the next char in cache.
      const logits =
        oneHot(cache.gather(index, 1), vocabSize).mul(1e9);
      return [logits, hiddenStates, cache];
    }

    next = nextFn;
    sampler = new TopKSampler({k: 5, temperature: 1.0});
  });

  it('stateless call', () => {
    function nextFn(
      prompt: Tensor, cache: Tensor, index: number
    ): [Tensor, Tensor, Tensor] {
      // Dummy hidden states.
      const hiddenStates = ones([batchSize, 5]);
      // Return a distribution favoring the first token in the vocab.
      const logits = oneHot(zeros([batchSize], 'int32'), vocabSize).mul(1e9);
      return [logits, hiddenStates, cache];
    }
    const prompt = fill([batchSize, length], charLookup['z']);
    const output = sampler.apply(nextFn, prompt, null, 5);

    test_util.expectArraysEqual(joinAsString(output), ['zzzzzaaaaaaa']);
  });

  it('stateful call', () => {
    const cacheChars = 'sequentially'.split('');
    const cache = tensor([cacheChars.map(c => charLookup[c])], null, 'int32');
    const prompt = fill([batchSize, length], charLookup['z']);
    const output = sampler.apply(next, prompt, cache);

    test_util.expectArraysEqual(joinAsString(output), ['sequentially']);
  });

  it('early stopping', () => {
    const cacheChars = 'sequentially'.split('');
    const cache = tensor([cacheChars.map(c => charLookup[c])], null, 'int32');
    const prompt = fill([batchSize, length], charLookup['z']);
    const output = sampler.apply(next, prompt, cache, 0, null, charLookup['t']);

    test_util.expectArraysEqual(joinAsString(output), ['sequentzzzzz']);
  });

  it('outputs in top k', () => {
    function nextFn(
      prompt: Tensor, cache: Tensor, index: number
    ): [Tensor, Tensor, Tensor] {
      // Dummy hidden states.
      const hiddenStates = ones([batchSize, 5]);
      // Return a distribution where each id is progressively less likely.
      const logits = range(vocabSize, 0, -1).expandDims(0);
      return [logits, hiddenStates, cache];

    }
    const prompt = fill([batchSize, length], charLookup['z']);
    const output = sampler.apply(nextFn, prompt);
    let outputIds = (output.arraySync() as number[][])[0];
    outputIds = [...new Set(outputIds)];

    console.log(outputIds);
    expect(
      outputIds.map(i => [0, 1, 2, 3, 4].includes(i)).includes(false)
    ).toBeFalse();
  });

  it('larger batch', () => {
    const cacheChars = 'sequentially'.split('');
    const cacheCharIds = cacheChars.map(c => charLookup[c]);
    const cache = tensor([cacheCharIds, cacheCharIds], null, 'int32');
    const prompt = fill([2, length], charLookup['z']);
    const output = sampler.apply(next, prompt, cache);

    test_util.expectArraysEqual(
      joinAsString(output),
      ['sequentially', 'sequentially']
    );
  });
});
