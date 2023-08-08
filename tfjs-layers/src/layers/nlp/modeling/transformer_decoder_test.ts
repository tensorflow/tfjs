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
import { Tensor, memory, randomUniform, randomUniformInt, serialization, zeros, zerosLike } from '@tensorflow/tfjs-core';

import { SymbolicTensor } from '../../../engine/topology';
import { input, model } from '../../../exports';
import { expectTensorsClose } from '../../../utils/test_utils';
import { Dense } from '../../core';
import { sliceUpdate } from '../utils';

import { TransformerDecoder } from './transformer_decoder';

describe('TransformerDecoder', () => {
  let originalTimeout: number;

  beforeAll(() => {
    // This test needs more time to finish the async fetch, adjusting
    // jasmine timeout for this test to avoid flakiness. See jasmine
    // documentation for detail:
    // https://jasmine.github.io/2.0/introduction.html#section-42
    originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });

  afterAll(() => jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout);

  describe('valid call', () => {
    function testValidCallWithoutCrossAttention(
        testcaseName: string, normalizeFirst: boolean) {
      it(`${testcaseName} without cross attention`, () => {
        const decoderInput = randomUniform([4, 6]);
        const decoder = new TransformerDecoder({
          intermediateDim: 4,
          numHeads: 2,
          normalizeFirst,
        });
        expect(() => decoder.apply(decoderInput)).not.toThrow();
      });
    }

    testValidCallWithoutCrossAttention('without_norm_first', false);
    testValidCallWithoutCrossAttention('with_norm_first', true);
  });

  it('invalid call', () => {
    const encoderInput = zeros([4, 6]);
    const decoderInput = zeros([4, 6]);

    // Without cross-attention.
    const decoder = new TransformerDecoder({intermediateDim: 4, numHeads: 2});
    decoder.apply(decoderInput);

    // Should raise ValueError if encoderInput is provided.
    expect(
      () => decoder.apply(decoderInput, {encoderSequence: encoderInput})
    ).toThrow();
  });

  it('error when invalid kernel initializer', () => {
    expect(() => new TransformerDecoder({
      intermediateDim: 4,
      numHeads: 2,
      dropout: 0.5,
      kernelInitializer: 'Invalid',
    })).toThrow();
  });

  it('one training step of transformer without cross attention', async () => {
    const decoderInput = input({shape: [4, 6]});
    const decoder = new TransformerDecoder({intermediateDim: 4, numHeads: 2});
    let outputs = decoder.apply(decoderInput);
    outputs = new Dense({
      units: 10, activation: 'softmax'}).apply(outputs) as SymbolicTensor;
    const tModel = model({inputs: decoderInput, outputs});

    const decoderSequence = randomUniform([2, 4, 6]);
    const label = randomUniformInt([2, 4, 1], 0, 10).asType('float32');

    tModel.compile({loss: 'sparseCategoricalCrossentropy', optimizer: 'adam'});
    const loss = tModel.trainOnBatch(decoderSequence, label);

    expect(await loss).toBeGreaterThan(0);
  });

  it('serialization round trip', () => {
    const testLayer = new TransformerDecoder({intermediateDim: 4, numHeads: 2});

    const config = testLayer.getConfig();
    const restored = TransformerDecoder.fromConfig(TransformerDecoder, config);

    // Initializers don't get serailized with customObjects.
    delete ((config['kernelInitializer'] as serialization.ConfigDict
      )['config'] as serialization.ConfigDict)['customObjects'];
    delete ((config['biasInitializer'] as serialization.ConfigDict
      )['config'] as serialization.ConfigDict)['customObjects'];

    expect(restored.getConfig()).toEqual(config);
  });

  it('does not leak memory', () => {
    const decoderInput = randomUniform([4, 6]);
    const decoder = new TransformerDecoder({intermediateDim: 4, numHeads: 2});
    // Initial apply to make sure layer is built.
    decoder.apply(decoderInput);

    const numTensors = memory().numTensors;
    decoder.apply(decoderInput);

    expect(memory().numTensors).toEqual(numTensors + 1);
  });

  it('cache call is correct', () => {
    const batchSize = 2;
    const seqLen = 5;
    const numHeads = 2;
    const headDim = 4;
    const hiddenDim = numHeads * headDim;

    const layer = new TransformerDecoder({intermediateDim: 4, numHeads});
    const dtype = 'float32';

    const x = randomUniform([batchSize, seqLen, hiddenDim], null, null, dtype);
    const cache = zeros([batchSize, 2, seqLen, numHeads, headDim], dtype);
    const outputs = zerosLike(x);

    layer.build(x.shape);
    const [noLoopOutputs, noLoopCache] = layer.callAndReturnCaches(
      x, {selfAttentionCache: cache, selfAttentionCacheUpdateIndex: 0});

    function call(outputs: Tensor, cache: Tensor) {
      for (let i = 0; i < seqLen; i++) {
        // Compute the rest tokens.
        const nextInput = x.slice([0, i, 0], [batchSize, 1, hiddenDim]);
        const [nextOutput, nextCache] = layer.callAndReturnCaches(
          nextInput,
          {
            selfAttentionCache: cache,
            selfAttentionCacheUpdateIndex: i
          }
        );
        outputs = sliceUpdate(outputs, [0, i, 0], nextOutput);
        cache = nextCache;
      }
      return [outputs, cache];
    }
    const [output, outputCache] = call(outputs, cache);

    expectTensorsClose(output, noLoopOutputs);
    expectTensorsClose(outputCache, noLoopCache);
  });

  // TODO(pforderique): Test mask propogation once supported.
});
