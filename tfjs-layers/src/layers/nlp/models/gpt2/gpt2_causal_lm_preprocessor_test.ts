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
 * Tests for GPT2 causal LM preprocessor layer.
 */

import { Tensor, tensor, tensor2d } from '@tensorflow/tfjs-core';

import { GPT2Tokenizer } from './gpt2_tokenizer';
import { expectTensorsClose } from '../../../../utils/test_utils';
import { GPT2TensorMap } from '../generative_task';
import { GPT2CausalLMPreprocessor } from './gpt2_causal_lm_preprocessor';

describe('GPT2CausalLMPreprocessorTest', () => {
  let vocabulary: Map<string, number>;
  let merges: string[];
  let preprocessor: GPT2CausalLMPreprocessor;

  beforeAll(() => {
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
    preprocessor = new GPT2CausalLMPreprocessor({
      tokenizer: new GPT2Tokenizer({vocabulary, merges}),
      sequenceLength: 8
    });
  });

  it('strings', () => {
    const inputData = tensor(['airplane at airport']);

    const [x, y, sw] = preprocessor.callAndPackArgs(
      inputData, {}
    ) as [GPT2TensorMap, Tensor, Tensor];

    expectTensorsClose(x.tokenIds, tensor([[6, 1, 3, 4, 2, 5, 6, 0]]));
    expectTensorsClose(
      x.paddingMask, tensor2d([[1, 1, 1, 1, 1, 1, 1, 0]], [1, 8], 'bool'));
    expectTensorsClose(y, tensor([[1, 3, 4, 2, 5, 6, 0, 0]]));
    expectTensorsClose(sw, tensor([[1, 1, 1, 1, 1, 1, 0, 0]]));
  });

  it('no start end token', () => {
    const inputData = tensor(Array<string>(4).fill('airplane at airport'));
    preprocessor = new GPT2CausalLMPreprocessor({
      tokenizer: new GPT2Tokenizer({vocabulary, merges}),
      sequenceLength: 8,
      addStartToken: false,
      addEndToken: false,
    });
    const expectedX = {
      tokenIds: tensor2d(Array<number[]>(4).fill([1, 3, 4, 2, 5, 0, 0, 0])),
      paddingMask: tensor2d(
        Array<number[]>(4).fill([1, 1, 1, 1, 1, 0, 0, 0]), [4, 8], 'bool'),
    };
    const expectedY = tensor2d(
      Array<number[]>(4).fill([3, 4, 2, 5, 0, 0, 0, 0]));
    const expectedSW = tensor2d(
      Array<number[]>(4).fill([1, 1, 1, 1, 0, 0, 0, 0]));

    const [x, y, sw] = preprocessor.callAndPackArgs(
      inputData, {}
    ) as [GPT2TensorMap, Tensor, Tensor];

    expectTensorsClose(x.tokenIds, expectedX.tokenIds);
    expectTensorsClose(x.paddingMask, expectedX.paddingMask);
    expectTensorsClose(y, expectedY);
    expectTensorsClose(sw, expectedSW);
  });

  it('tokenize labeled batch', () => {
    let inputData = tensor(Array<string>(4).fill('airplane at airport'));
    let y = tensor([1, 1, 1, 1]);  // Ignored.
    let sw = tensor([1., 1., 1., 1.]);  // Ignored.
    const expectedX = {
      tokenIds: tensor2d(Array<number[]>(4).fill([6, 1, 3, 4, 2, 5, 6, 0])),
      paddingMask: tensor2d(
        Array<number[]>(4).fill([1, 1, 1, 1, 1, 1, 1, 0]), [4, 8], 'bool'),
    };

    const expectedY = tensor2d(
      Array<number[]>(4).fill([1, 3, 4, 2, 5, 6, 0, 0]));
    const expectedSW = tensor2d(
      Array<number[]>(4).fill([1, 1, 1, 1, 1, 1, 0, 0]));

    let x: GPT2TensorMap;
    [x, y, sw] = preprocessor.callAndPackArgs(
      inputData, {y, sampleWeight: sw}
    ) as [GPT2TensorMap, Tensor, Tensor];

    expectTensorsClose(x.tokenIds, expectedX.tokenIds);
    expectTensorsClose(x.paddingMask, expectedX.paddingMask);
    expectTensorsClose(y, expectedY);
    expectTensorsClose(sw, expectedSW);
  });

  it('generate preprocess', () => {
    const inputData = tensor(['airplane at airport']);
    const x = preprocessor.generatePreprocess(inputData);

    expectTensorsClose(x.tokenIds, tensor2d([[6, 1, 3, 4, 2, 5, 0, 0]]));
    expectTensorsClose(x.paddingMask, tensor2d(
      [[1, 1, 1, 1, 1, 1, 0, 0]], [1, 8], 'bool'));
  });

  it('generate postprocess', () => {
    const inputData = {
      tokenIds: tensor2d([[6, 1, 3, 4, 2, 5, 0, 0]]),
      paddingMask: tensor2d([[1, 1, 1, 1, 1, 1, 0, 0]], [1, 8], 'bool'),
    };

    const x = preprocessor.generatePostprocess(inputData);

    expectTensorsClose(x, tensor(['airplane at airport']));
  });

  // TODO(pforderique): Test serialization.
});
