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
 *  Utility functions for `TransformerDecoder`.
 */

/* Original source: keras_nlp/layers/modeling/transformer_layer_utils.py */
import { Tensor, add, expandDims, tensor, tidy } from '@tensorflow/tfjs-core';

import { ValueError } from '../../../errors';

function checkMasksShapes(
    inputs: Tensor, paddingMask: Tensor, attentionMask: Tensor): void {
  if (paddingMask != null) {
    if (paddingMask.shape.length !==2) {
      throw new ValueError(
        '`paddingMask` should have shape ' +
        `[batchSize, targetLength]. Received shape ${paddingMask.shape}.`
      );
    }
  }
  if (attentionMask != null) {
    if (attentionMask.shape.length !== 3) {
      throw new ValueError(
        '`attentionMask` should have shape ' +
        `[batchSize, targetLength, sourceLength]. ` +
        `Received shape ${attentionMask.shape}.`
      );
    }
  }
}

/**
 * Compute a causal attention mask for a transformer decoder.
 *
 * @param batchSize batch size for the mask.
 * @param inputLength the length of key/value tensors in the attention layer.
 * @param outputLength the length of query tensor in the attention layer.
 * @param cacheIndex the current index for cached generation. If passed, the
 *  query sequence will be considered to start at `cacheIndex` rather than zero.
 *  For example, a casual mask with `outputLength=1` and `cacheIndex=5` would
 *  allow the query tensor to attend to the first five positions of the
 *  key/value tensors.
 *
 * @returns a causal attention mask with shape
 *  `[batchSize, outputLength, inputLength]` that can be passed to a attention
 *  layer.
 */
export function computeCausalMask(
    batchSize: number,
    inputLength: number,
    outputLength: number,
    cacheIndex = 0
  ): Tensor {
  return tidy(() => {
    const i = add(
      expandDims(Array.from({length: outputLength}, (_, i) => i), 1),
      cacheIndex,
    );
    const j = tensor(Array.from({length: inputLength}, (_, i) => i));
    const mask = i.greaterEqual(j).cast('int32').expandDims(0);
    return mask.broadcastTo([batchSize, outputLength, inputLength]);
  });
}

/**
 * Merge the padding mask with a customized attention mask.
 *
 * @param inputs the input sequence.
 * @param paddingMask the 1D padding mask, of shape
 *          [batchSize, sequenceLength].
 * @param attentionMask the 2D customized mask, of shape
 *          [batchSize, sequenceLength, sequence2_length].
 * @returns
 *  A merged 2D mask or null. If only `paddingMask` is provided, the
 *  returned mask is paddingMask with one additional axis.
 */
export function mergePaddingAndAttentionMask(
    inputs: Tensor, paddingMask: Tensor, attentionMask: Tensor): Tensor {
  return tidy(() => {
    checkMasksShapes(inputs, paddingMask, attentionMask);
    let mask: Tensor;
    if (paddingMask != null) {
      // Add an axis for broadcasting, the attention mask should be 2D
      // (not including the batch axis).
      mask = paddingMask.expandDims(1).cast('int32');
    }
    if (attentionMask != null) {
      attentionMask = attentionMask.cast('int32');
      if (mask == null) {
        return attentionMask;
      } else {
        return mask.minimum(attentionMask);
      }
    }
    return mask;
  });
}
