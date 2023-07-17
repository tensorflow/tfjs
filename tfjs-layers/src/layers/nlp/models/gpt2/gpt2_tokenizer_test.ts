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
 * Unit Tests for GPT2Tokenizer.
 */

import { tensor } from '@tensorflow/tfjs-core';

import { expectTensorsClose } from '../../../../utils/test_utils';
import { GPT2Tokenizer } from './gpt2_tokenizer';

describe('GPT2Tokenizer', () => {
  let vocabulary: Map<string, number>;
  let merges: string[];
  let tokenizer: GPT2Tokenizer;

  beforeEach(() => {
    vocabulary = new Map([
      ['<|endoftext|>', 0],
      ['Ġair', 1],
      ['plane', 2],
      ['Ġat', 3],
      ['port', 4],
      ['Ġkoh', 5],
      ['li', 6],
      ['Ġis', 7],
      ['Ġthe', 8],
      ['Ġbest', 9],
    ]);

    merges = ['Ġ a', 'Ġ t', 'Ġ k', 'Ġ i', 'Ġ b', 'Ġa i', 'p l', 'n e'].concat(
      ['Ġa t', 'p o', 'r t', 'o h', 'l i', 'Ġi s', 'Ġb e', 's t'],
      ['Ġt h', 'Ġai r', 'pl a', 'Ġk oh', 'Ġth e', 'Ġbe st', 'po rt'],
      ['pla ne']);

    tokenizer = new GPT2Tokenizer({vocabulary, merges});
  });

  it('tokenize', () => {
    const inputData = tensor([' airplane at airport']);
    const expectedOutput = [tensor([1, 2, 3, 1, 4])];

    const output = tokenizer.tokenize(inputData);

    expect(output.length).toBe(1);
    expectTensorsClose(output[0], expectedOutput[0]);
  });

  it('tokenize end token', () => {
    const inputData = tensor([' airplane at airport<|endoftext|>']);
    const expectedOutput = [tensor([1, 2, 3, 1, 4, 0])];

    const output = tokenizer.tokenize(inputData);

    expect(output.length).toBe(1);
    expectTensorsClose(output[0], expectedOutput[0]);
  });

  it('tokenize batch', () => {
    const inputData = tensor([' airplane at airport', ' kohli is the best']);
    const expectedOutput = [tensor([1, 2, 3, 1, 4]), tensor([5, 6, 7, 8, 9])];

    const output = tokenizer.tokenize(inputData);

    expect(output.length).toBe(2);
    expectTensorsClose(output[0], expectedOutput[0]);
    expectTensorsClose(output[1], expectedOutput[1]);
  });

  it('detokenize', () => {
    const inputData = [tensor([1, 2, 3, 1, 4])];
    const expectedOutput = tensor([' airplane at airport']);

    const output = tokenizer.detokenize(inputData);

    expectTensorsClose(output, expectedOutput);
  });

  it('vocabulary size', () => {
    expect(tokenizer.vocabularySize).toBe(10);
  });

  it('errors with missing special tokens', () => {
    vocabulary = new Map([['a', 1], ['b', 2], ['c', 3]]);
    merges = [];
    expect(() => new GPT2Tokenizer({vocabulary, merges})).toThrowError();
  });
});
