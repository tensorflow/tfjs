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
 * Unit Tests for CachedMultiHeadAttention layer.
 */
import { Tensor, linalg, memory, ones, randomUniform, zeros, zerosLike } from '@tensorflow/tfjs-core';

import { expectTensorsClose } from '../../../utils/test_utils';
import { sliceUpdate } from '../utils';

import { CachedMultiHeadAttention } from './cached_multihead_attention';

describe('CachedMultiHeadAttention', () => {
  it('valid call', () => {
    const layer = new CachedMultiHeadAttention({numHeads: 2, keyDim: 4});
    const query = randomUniform([2, 2, 8]);

    expect(() => layer.call(query, {value: query})).not.toThrow();
  });

  it('cache call is correct', () => {
    const batchSize = 2;
    const seqLen = 5;
    const numHeads = 2;
    const keyDim = 4;
    const hiddenDim = numHeads * keyDim;
    const inputShape = [batchSize, seqLen, hiddenDim];

    const x = randomUniform(inputShape);
    const inputCache = zeros([batchSize, 2, seqLen, numHeads, keyDim]);
    // Use a causal mask.
    const mask = linalg.bandPart(ones([seqLen, seqLen]), -1, 0);
    const outputs = zerosLike(x);

    const layer = new CachedMultiHeadAttention({numHeads, keyDim});
    const [noLoopOutputs, noLoopCache] = layer.callAndReturnCache(
      x,
      {
        value: x,
        cache: inputCache,
        cacheUpdateIndex: 0,
        attentionMask: mask
      }
    );

    function call(outputs: Tensor, cache: Tensor) {
      for (let i = 0; i < seqLen; i++) {
        // Compute the rest tokens.
        const nextInput = x.slice([0, i, 0], [batchSize, 1, hiddenDim]);
        const nextMask = mask.slice([i, 0], [1, seqLen]);
        const [nextOutput, nextCache] = layer.callAndReturnCache(
          nextInput,
          {
            value: nextInput,
            cache,
            cacheUpdateIndex: i,
            attentionMask: nextMask,
          }
        );
        outputs = sliceUpdate(outputs, [0, i, 0], nextOutput);
        cache = nextCache;
      }
      return [outputs, cache];
    }
    const [output, outputCache] = call(outputs, inputCache);

    expectTensorsClose(output, noLoopOutputs);
    expectTensorsClose(outputCache, noLoopCache);
  });

  it('does not leak memory', () => {
    const layer = new CachedMultiHeadAttention({numHeads: 2, keyDim: 2});
    const query = ones([1, 4, 8]);
    // Initial call that builds sublayers and necessary tensors.
    layer.call(query, {value: query});

    const numTensors = memory().numTensors;
    layer.call(query, {value: query});

    expect(memory().numTensors).toEqual(numTensors + 1);
  });
  // TODO(pforderique): Test serialization.
});
