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
 *  Position embedding implementation based on `tf.layers.Layer`.
 */

/* Original source: keras_nlp/layers/modeling/position_embedding.py */
import { Tensor, Tensor1D, Tensor2D, serialization } from '@tensorflow/tfjs-core';

import { Shape } from '../../../keras_format/common';
import { Layer, LayerArgs } from '../../../engine/topology';
import { NotImplementedError } from '../../../errors';
import { InitializerIdentifier } from '../../../initializers';

export declare interface PositionEmbeddingArgs extends LayerArgs {
  /**
   * Integer. The maximum length of the dynamic sequence.
   */
  sequenceLength: number;

  /**
   * The initializer to use for the embedding weights.
   * Defaults to `"glorotUniform"`.
   */
  initializer?: InitializerIdentifier;
}

export declare interface PositionEmbeddingOptions {
  /**
   * Integer. Index to start the position embeddings at.
   * Defaults to 0.
   */
  startIndex?: number;
}

/**
 * A layer which learns a position embedding for input sequences.
 *
 * This class assumes that in the input tensor, the last dimension corresponds
 * to the features, and the dimension before the last corresponds to the
 * sequence.
 *
 * Examples:
 *
 * Called directly on input.
 * ```js
 * const layer = new PositionEmbedding({sequenceLength=10});
 * layer.call(tf.zeros([8, 10, 16]));
 * ```
 *
 * Combine with a token embedding.
 * ```js
 * const seqLength = 50;
 * const vocabSize = 5000;
 * const embedDim = 128;
 * const inputs = tf.input({shape: [seqLength]});
 * const tokenEmbeddings = tf.layers.embedding({
 *    inputDim=vocabSize, outputDim=embedDim
 * }).apply(inputs)
 * const positionEmbeddings = new PositionEmbedding({
 *     sequenceLength: seqLength
 * }).apply(tokenEmbeddings)
 * const outputs = tf.add(tokenEmbeddings, positionEmbeddings);
 * ```
 *
 * Reference:
 *  - [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
 */
export class PositionEmbedding extends Layer {
  /** @nocollapse */
  static readonly className = 'PositionEmbedding';

  constructor(args: PositionEmbeddingArgs) {
    super(args);

    throw new NotImplementedError('PositionEmbedding not implemented yet.');
  }

  override getConfig(): serialization.ConfigDict {
    throw new NotImplementedError('Not implemented yet.');
  }

  override build(inputShape: Shape | Shape[]): void {
    throw new NotImplementedError('Not implemented yet.');
  }

  override call(
    inputs: Tensor|Tensor[],
    kwargs: PositionEmbeddingOptions={startIndex: 0}
  ): Tensor1D|Tensor2D {
    throw new NotImplementedError('Not implemented yet.');
  }
}
serialization.registerClass(PositionEmbedding);
