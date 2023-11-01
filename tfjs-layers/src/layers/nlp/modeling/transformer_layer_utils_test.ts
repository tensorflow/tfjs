/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * Unit Tests for Transformer Decoder.
 */
import { memory, randomUniform, tensor } from '@tensorflow/tfjs-core';

import { expectTensorsClose } from '../../../utils/test_utils';

import { computeCausalMask, mergePaddingAndAttentionMask } from './transformer_layer_utils';

describe('TransformerLayerUtils', () => {
  it('compute causal mask', () => {
    const mask = computeCausalMask(1, 2, 2);
    expect(mask.arraySync()).toEqual([[[1, 0], [1, 1]]]);
  });

  it('merge padding and attention mask', () => {
    const paddingMask = tensor([[1, 1, 0]]);
    const attentionMask = tensor([[[0, 0, 1], [0, 1, 0], [1, 0, 0]]]);
    const inputs = randomUniform([1, 3, 2]);

    const mergedMask =
      mergePaddingAndAttentionMask(inputs, paddingMask, attentionMask);

    expectTensorsClose(
      mergedMask,
      tensor([[[0, 0, 0], [0, 1, 0], [1, 0, 0]]], null, 'int32')
    );
  });

  it('bad mask shapes', () => {
    let paddingMask = tensor([[[1, 1, 0], [1, 0, 0]]]);
    let attentionMask = tensor([[[0, 0, 1], [0, 1, 0], [1, 0, 0]]]);
    let inputs = randomUniform([1, 3, 2]);

    expect(
      () => mergePaddingAndAttentionMask(inputs, paddingMask, attentionMask)
    ).toThrow();

    paddingMask = tensor([[1, 1, 0]]);
    attentionMask = tensor([[0, 0, 1], [1, 0, 0]]);
    inputs = randomUniform([1, 3, 2]);

    expect(
      () => mergePaddingAndAttentionMask(inputs, paddingMask, attentionMask)
    ).toThrow();
  });

  it('does not leak memory', () => {
    // computeCausalMask
    let numTensorsBefore = memory().numTensors;
    computeCausalMask(1, 2, 2);
    expect(memory().numTensors).toEqual(numTensorsBefore + 1);

    // mergePaddingAndAttentionMask
    const paddingMask = tensor([[1, 1, 0]]);
    const attentionMask = tensor([[[0, 0, 1], [0, 1, 0], [1, 0, 0]]]);
    const inputs = randomUniform([1, 3, 2]);
    numTensorsBefore = memory().numTensors;
    mergePaddingAndAttentionMask(inputs, paddingMask, attentionMask);
    expect(memory().numTensors).toEqual(numTensorsBefore + 1);
  });
  // TODO(pforderique): Test serialization.
});
