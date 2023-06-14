/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit Tests for Tokenizer Layers.
 */

import { Tensor1D, tensor1d } from '@tensorflow/tfjs-core';

import { WhiteSpaceTokenizer } from './tokenizers';
import { expectTensorsClose } from '../../../src/utils/test_utils';

describe('White Space Tokenizer', () => {
  const tokenizer = new WhiteSpaceTokenizer();

  it('tokenize', () => {
    const inputData = tensor1d(["the quick brown fox"]);
    const expectedOutput = [tensor1d(["the", "quick", "brown", "fox"])];

    const tokenizeOutput = tokenizer.tokenize(inputData);
    const callOutput = tokenizer.call(inputData) as Tensor1D[];

    expect(tokenizeOutput.length).toBe(1);
    expectTensorsClose(tokenizeOutput[0], expectedOutput[0]);

    expect(callOutput.length).toBe(1);
    expectTensorsClose(callOutput[0], expectedOutput[0]);
  });

  it('detokenize', () => {
    const inputData = [tensor1d(["the", "quick", "brown", "fox"])];
    const expectedOutput = tensor1d(["the quick brown fox"]);

    const detokenizeOutput = tokenizer.detokenize(inputData);
    const callOutput = tokenizer.call(inputData, {mode: 'detokenize'}) as Tensor1D;

    expectTensorsClose(detokenizeOutput, expectedOutput);
    expectTensorsClose(callOutput, expectedOutput);
  });

  it('detokenize(tokenize) composition', () => {
    const inputData = tensor1d(["the quick brown fox"]);

    expectTensorsClose(
      tokenizer.detokenize(tokenizer.tokenize(inputData)), inputData);
  });
});
