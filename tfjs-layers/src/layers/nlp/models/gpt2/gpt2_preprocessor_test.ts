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
 * Unit Tests for GPT2Preprocessor.
 */

import { NamedTensorMap, Tensor, memory, serialization, tensor, tensor2d } from '@tensorflow/tfjs-core';

import { GPT2Preprocessor } from './gpt2_preprocessor';
import { GPT2Tokenizer } from './gpt2_tokenizer';
import { expectTensorsClose } from '../../../../utils/test_utils';

describe('GPT2Preprocessor', () => {
  let vocabulary: Map<string, number>;
  let merges: string[];
  let preprocessor: GPT2Preprocessor;

  beforeEach(() => {
    vocabulary = new Map([
      ['!', 0],
      ['air', 1],
      ['Ġair', 2],
      ['plane', 3],
      ['Ġat', 4],
      ['port', 5],
      ['<|endoftext|>', 6],
    ]);

    merges = ['Ġ a', 'Ġ t', 'Ġ i', 'Ġ b', 'a i', 'p l', 'n e'].concat(
      ['Ġa t', 'p o', 'r t', 'Ġt h', 'ai r', 'pl a', 'po rt'],
      ['Ġai r', 'Ġa i', 'pla ne']
    );
    preprocessor = new GPT2Preprocessor({
      tokenizer: new GPT2Tokenizer({vocabulary, merges}),
      sequenceLength: 8
    });
  });

  it('tokenize', () => {
    const inputData = tensor(['airplane at airport']);

    const output =
      preprocessor.callAndPackArgs(inputData, {}) as NamedTensorMap;

    expectTensorsClose(output.tokenIds, tensor2d([[6, 1, 3, 4, 2, 5, 6, 0]]));
    expectTensorsClose(
      output.paddingMask, tensor2d([[1, 1, 1, 1, 1, 1, 1, 0]], [1, 8], 'bool'));
  });

  it('no start end token', () => {
    const inputData = tensor(Array<string>(4).fill('airplane at airport'));
    preprocessor = new GPT2Preprocessor({
      tokenizer: new GPT2Tokenizer({vocabulary, merges}),
      sequenceLength: 8,
      addStartToken: false,
      addEndToken: false,
    });
    const expectedOutput = {
      tokenIds: tensor2d(Array<number[]>(4).fill([1, 3, 4, 2, 5, 0, 0, 0])),
      paddingMask: tensor2d(
        Array<number[]>(4).fill([1, 1, 1, 1, 1, 0, 0, 0]), [4, 8], 'bool'),
    };

    const output =
      preprocessor.callAndPackArgs(inputData, {}) as NamedTensorMap;

    expectTensorsClose(output.tokenIds, expectedOutput.tokenIds);
    expectTensorsClose(output.paddingMask, expectedOutput.paddingMask);
  });

  it('tokenize labeled batch', () => {
    const inputData = tensor(Array<string>(4).fill('airplane at airport'));
    const yIn = tensor([1, 1, 1, 1]);
    const swIn = tensor([1., 1., 1., 1.]);
    const expectedX = {
      tokenIds: tensor2d(Array<number[]>(4).fill([6, 1, 3, 4, 2, 5, 6, 0])),
      paddingMask: tensor2d(
        Array<number[]>(4).fill([1, 1, 1, 1, 1, 1, 1, 0]), [4, 8], 'bool'),
    };

    const output = preprocessor.callAndPackArgs(
      inputData, {y: yIn, sampleWeight: swIn}
    ) as [NamedTensorMap, Tensor, Tensor];

    expectTensorsClose(output[0].tokenIds, expectedX.tokenIds);
    expectTensorsClose(output[0].paddingMask, expectedX.paddingMask);
    expectTensorsClose(output[1], yIn);
    expectTensorsClose(output[2], swIn);
  });

  it('sequence length override', () => {
    const inputData = tensor(['airplane at airport']);

    const output = preprocessor.callAndPackArgs(
      inputData, {sequenceLength: 4}
    ) as NamedTensorMap;

    expectTensorsClose(output.tokenIds, tensor2d([[6, 1, 3, 6]]));
  });

  it('does not leak memory', () => {
    const inputData = tensor(['airplane at airport']);

    const numTensorsBefore = memory().numTensors;
    preprocessor.callAndPackArgs(inputData, {sequenceLength: 4});
    const numTensorsAfter = memory().numTensors;
    expect(numTensorsAfter).toEqual(numTensorsBefore + 2);
  });

  it('serialization round-trip', () => {
    const reserialized = GPT2Preprocessor.fromConfig(
      GPT2Preprocessor, preprocessor.getConfig());

    const originalConfig = preprocessor.getConfig();
    const reserializedConfig = reserialized.getConfig();

    // TODO(pforderique): Verify any tokenizer name consistency issues.
    delete ((originalConfig['tokenizer'] as serialization.ConfigDict
      )['config'] as serialization.ConfigDict) ['name'];
    delete ((reserializedConfig['tokenizer'] as serialization.ConfigDict
      )['config'] as serialization.ConfigDict) ['name'];

    expect(reserializedConfig).toEqual(originalConfig);
  });
});
