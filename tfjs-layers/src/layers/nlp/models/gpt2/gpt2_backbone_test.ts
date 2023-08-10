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

import { Tensor, memory, ones } from '@tensorflow/tfjs-core';
import { GPT2Backbone } from './gpt2_backbone';

/**
 *  Tests for GPT-2 backbone models.
 */

describe('GPT2Backbone', () => {
  let backbone: GPT2Backbone;
  // let inputBatch: {[name: string]: Tensor};
  let inputBatch: Tensor[];

  beforeAll(() => {
    backbone = new GPT2Backbone({
      vocabularySize: 10,
      numLayers: 2,
      numHeads: 2,
      hiddenDim: 2,
      intermediateDim: 4,
      maxSequenceLength: 5,
    });
    inputBatch = [
      ones([2, 5], 'int32'),  // tokenIds
      ones([2, 5], 'int32'),  // paddingMask
    ];
  });

  it('call', () => {
    expect(() => backbone.apply(inputBatch)).not.toThrow();
  });

  it('token embedding', () => {
    const output = backbone.tokenEmbedding.apply(inputBatch[0]) as Tensor;
    expect(output.shape).toEqual([2, 5, 2]);
  });

  it('name', () => {
    // Check default name passed through.
    expect(backbone.name).toMatch('gpt2_backbone');
  });

  it('variable sequence length', () => {
    let inputData: Tensor[];
    for (const seqLength of [2, 3, 4]) {
      inputData = [
        ones([2, seqLength], 'int32'),  // tokenIds
        ones([2, seqLength], 'int32'),  // paddingMask
      ];
      expect(() => backbone.apply(inputData)).not.toThrow();
    }
  });

  it('predict', () => {
    expect(() => backbone.predict(inputBatch)).not.toThrow();
  });

  it('does not leak memory', () => {
    const numTensors = memory().numTensors;
    backbone.apply(inputBatch);

    expect(memory().numTensors).toEqual(numTensors + 1);
  });
});
