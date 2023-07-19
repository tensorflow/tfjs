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
 * Unit Tests for Tokenizer Layers.
 */

import { Tensor, memory, tensor, test_util } from '@tensorflow/tfjs-core';

import { BytePairTokenizer, Tokenizer } from './tokenizers';
import { expectTensorsClose } from '../../utils/test_utils';

class SimpleTokenizer extends Tokenizer {
  /** @nocollapse */
  static className = 'SimpleTokenizer';

  tokenize(inputs: Tensor): Tensor[] {
    const stringInputs = inputs.dataSync() as unknown as string[];
    return stringInputs.map(input => tensor(input.split(' ')));
  }

  override detokenize(inputs: Tensor[]): Tensor {
    const stringInputs = inputs.map(
      input => input.dataSync() as unknown as string[]);
    return tensor(stringInputs.map(str => str.join(' ')));
  }
}

describe('Tokenizer', () => {
  let tokenizer: SimpleTokenizer;

  beforeEach(() => {
    tokenizer = new SimpleTokenizer();
  });

  it('tokenize', () => {
    const inputData = tensor(['the quick brown fox']);
    const expectedOutput = [tensor(['the', 'quick', 'brown', 'fox'])];

    const tokenizeOutput = tokenizer.tokenize(inputData);
    const callOutput = tokenizer.call(inputData) as Tensor[];

    expect(tokenizeOutput.length).toBe(1);
    expectTensorsClose(tokenizeOutput[0], expectedOutput[0]);

    expect(callOutput.length).toBe(1);
    expectTensorsClose(callOutput[0], expectedOutput[0]);
  });

  it('detokenize', () => {
    const inputData = [tensor(['the', 'quick', 'brown', 'fox'])];
    const expectedOutput = tensor(['the quick brown fox']);

    const detokenizeOutput = tokenizer.detokenize(inputData);
    const callOutput = tokenizer.call(
      inputData, {mode: 'detokenize'}) as Tensor;

    expectTensorsClose(detokenizeOutput, expectedOutput);
    expectTensorsClose(callOutput, expectedOutput);
  });

  it('detokenize(tokenize) composition', () => {
    const inputData = tensor(['the quick brown fox']);

    expectTensorsClose(
      tokenizer.detokenize(tokenizer.tokenize(inputData)), inputData);
  });
});

describe('BytePairTokenizer', () => {
  it('gets correct set up', () => {
    const vocabulary = new Map([['butter', 1], ['fly', 2]]);
    const merges = ['b u', 't t', 'e r', 'bu tt', 'butt er', 'f l', 'fl y'];
    const tokenizer = new BytePairTokenizer({vocabulary, merges});
    const config = tokenizer.getConfig();

    expect(tokenizer.vocabulary).toEqual(['butter', 'fly']);
    expect(tokenizer.vocabularySize).toEqual(2);
    expect(tokenizer.idToToken(1)).toEqual('butter');
    expect(tokenizer.idToToken(3)).toEqual(undefined);
    expect(tokenizer.tokenToId('butter')).toEqual(1);
    test_util.expectArraysEqual(config.merges as string[], merges);
  });

  it('tokenize with special tokens', () => {
    const vocabulary = new Map([['sp', 0], ['s', 1], ['p', 2]]);
    const merges = ['s p'];
    let tokenizer = new BytePairTokenizer({
      vocabulary,
      merges,
      unsplittableTokens: ['s', 'p'],
    });
    const inputData = tensor(['sp']);
    const expectedOutput = [tensor([1, 2])];

    const tokenizeOutput = tokenizer.tokenize(inputData);
    const callOutput = tokenizer.call(inputData) as Tensor[];

    expect(tokenizeOutput.length).toBe(1);
    expectTensorsClose(tokenizeOutput[0], expectedOutput[0]);

    expect(callOutput.length).toBe(1);
    expectTensorsClose(callOutput[0], expectedOutput[0]);

    // If not setting special tokens, "sp" is one token.
    tokenizer = new BytePairTokenizer({
      vocabulary,
      merges,
    });
    const output = tokenizer.tokenize(inputData);
    expect(output.length).toBe(1);
    expectTensorsClose(output[0], tensor([0]));
  });

  it('tokenize with merges', () => {
    const vocabulary = new Map([
      ['br', 0], ['wn', 1], ['ck', 2], ['b', 3], ['r', 4], ['o', 5], ['w', 6],
      ['n', 7], ['.', 8], ['l', 9], ['a', 10], ['c', 11], ['d', 12]
    ]);
    const merges = ['b r', 'w n', 'c k'];
    const tokenizer = new BytePairTokenizer({vocabulary, merges});
    const inputData = tensor(['brown.', 'black.']);
    const expectedOutput = [tensor([0, 5, 1, 8]), tensor([3, 9, 10, 2, 8])];

    const tokenizeOutput = tokenizer.tokenize(inputData);
    const callOutput = tokenizer.call(inputData) as Tensor[];

    expect(tokenizeOutput.length).toBe(2);
    expectTensorsClose(tokenizeOutput[0], expectedOutput[0]);
    expectTensorsClose(tokenizeOutput[1], expectedOutput[1]);

    expect(callOutput.length).toBe(2);
    expectTensorsClose(callOutput[0], expectedOutput[0]);
    expectTensorsClose(callOutput[1], expectedOutput[1]);
  });

  it('tokenize with prefix space', () => {
    const vocabulary = new Map([
      ['br', 0], ['wn', 1], ['ck', 2], ['b', 3], ['r', 4], ['o', 5], ['w', 6],
      ['n', 7], ['.', 8], ['l', 9], ['a', 10], ['c', 11], ['d', 12]
    ]);
    const merges = ['b r', 'w n', 'c k'];
    const tokenizer = new BytePairTokenizer({
      vocabulary,
      merges,
      addPrefixSpace: true,
    });
    const inputData = tensor(['brown.', 'black.']);
    const expectedOutput = [
      tensor([-1, 0, 5, 1, 8]),
      tensor([-1, 3, 9, 10, 2, 8])
    ];

    const tokenizeOutput = tokenizer.tokenize(inputData);

    expect(tokenizeOutput.length).toBe(2);
    expectTensorsClose(tokenizeOutput[0], expectedOutput[0]);
    expectTensorsClose(tokenizeOutput[1], expectedOutput[1]);
  });

  it('tokenize with whitespace split', () => {
    const vocabulary = new Map([['\n\n', 0], ['\n', 1], [' ', 2], ['s', 3]]);
    const merges = ['\n \n'];
    const tokenizer = new BytePairTokenizer({vocabulary, merges});
    const inputData = tensor(['\n\n\n  s']);
    const expectedOutput = [tensor([-1, -1, -1, -1, -1, 3])];

    const tokenizeOutput = tokenizer.tokenize(inputData);

    expect(tokenizeOutput.length).toBe(1);
    expectTensorsClose(tokenizeOutput[0], expectedOutput[0]);
  });

  it('tokenize with unsplittable tokens', () => {
    const vocabulary = new Map([['<|end|>', 0], ['butter', 1], ['fly', 2]]);
    const merges = ['b u', 't t', 'e r', 'bu tt', 'butt er', 'f l', 'fl y'];
    const tokenizer = new BytePairTokenizer({
      vocabulary, merges, unsplittableTokens: ['<|end|>']});
    const inputData = tensor(['butterfly<|end|>']);
    const expectedOutput = [tensor([1, 2, 0])];

    const tokenizeOutput = tokenizer.tokenize(inputData);

    expect(tokenizeOutput.length).toBe(1);
    expectTensorsClose(tokenizeOutput[0], expectedOutput[0]);
  });

  it('tokenize with sequence length', () => {
    const vocabulary = new Map([['butter', 1], ['fly', 2]]);
    const merges = ['b u', 't t', 'e r', 'bu tt', 'butt er', 'f l', 'fl y'];
    const tokenizer = new BytePairTokenizer({
      vocabulary, merges, sequenceLength: 2});
    const inputData = tensor(['butterfly', 'butter', 'butterflybutter']);
    const expectedOutput = [tensor([1, 2]), tensor([1, 0]), tensor([1, 2])];

    const tokenizeOutput = tokenizer.tokenize(inputData);

    expect(tokenizeOutput.length).toBe(3);
    expectTensorsClose(tokenizeOutput[0], expectedOutput[0]);
    expectTensorsClose(tokenizeOutput[1], expectedOutput[1]);
    expectTensorsClose(tokenizeOutput[2], expectedOutput[2]);
  });

  it('tokenize does not leak memory', () => {
    const vocabulary = new Map([['butter', 1], ['fly', 2]]);
    const merges = ['b u', 't t', 'e r', 'bu tt', 'butt er', 'f l', 'fl y'];
    const tokenizer = new BytePairTokenizer({vocabulary, merges});
    const inputData = tensor(['butterfly']);

    const numTensorsBefore = memory().numTensors;
    tokenizer.tokenize(inputData);
    const numTensorsAfter = memory().numTensors;

    expect(numTensorsAfter).toEqual(numTensorsBefore + 1);
  });

  it('detokenize with multiple inputs', () => {
    const vocabulary = new Map([['butter', 1], ['fly', 2]]);
    const merges = ['b u', 't t', 'e r', 'bu tt', 'butt er', 'f l', 'fl y'];
    const tokenizer = new BytePairTokenizer({vocabulary, merges});
    const inputData = tensor(['butterfly', 'butter']);

    const output = tokenizer.detokenize(tokenizer.tokenize(inputData));

    expectTensorsClose(output, inputData);
  });
});
