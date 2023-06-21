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

import { Tensor, tensor } from '@tensorflow/tfjs-core';

import { Tokenizer } from './tokenizers';
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
