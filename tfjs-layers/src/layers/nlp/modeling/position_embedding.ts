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
import { Tensor, serialization, tidy } from '@tensorflow/tfjs-core';

import { Shape } from '../../../keras_format/common';
import { Layer, LayerArgs } from '../../../engine/topology';
import { ValueError } from '../../../errors';
import { Initializer, InitializerIdentifier, getInitializer, serializeInitializer } from '../../../initializers';
import { getExactlyOneTensor } from '../../../utils/types_utils';
import { LayerVariable } from '../../../variables';

export declare interface PositionEmbeddingArgs extends LayerArgs {
  /**
   * Integer. The maximum length of the dynamic sequence.
   */
  sequenceLength: number;

  /**
   * The initializer to use for the embedding weights.
   * Defaults to `"glorotUniform"`.
   */
  initializer?: Initializer|InitializerIdentifier;
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
 *     inputDim=vocabSize, outputDim=embedDim
 * }).apply(inputs);
 * const positionEmbeddings = new PositionEmbedding({
 *     sequenceLength: seqLength
 * }).apply(tokenEmbeddings);
 * const outputs = tf.add(tokenEmbeddings, positionEmbeddings);
 * ```
 *
 * Reference:
 *  - [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
 */
export class PositionEmbedding extends Layer {
  /** @nocollapse */
  static readonly className = 'PositionEmbedding';
  private sequenceLength: number;
  private initializer: Initializer;
  protected positionEmbeddings: LayerVariable;

  constructor(args: PositionEmbeddingArgs) {
    super(args);
    if (args.sequenceLength == null) {
      throw new ValueError(
        '`sequenceLength` must be an Integer, received `null`.');
    }
    this.sequenceLength = args.sequenceLength;
    this.initializer = getInitializer(args.initializer || 'glorotUniform');
  }

  override getConfig(): serialization.ConfigDict {
    const config = {
      'sequenceLength': this.sequenceLength,
      'initializer': serializeInitializer(this.initializer),
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  override build(inputShape: Shape): void {
    const featureSize = inputShape[inputShape.length - 1];
    this.positionEmbeddings = this.addWeight(
      'embeddings',
      [this.sequenceLength, featureSize],
      null,
      this.initializer,
      null,
      true
    );
    super.build(inputShape);
  }

  override call(
    inputs: Tensor|Tensor[],
    kwargs?: PositionEmbeddingOptions
  ): Tensor {
    return tidy(() => {
      kwargs.startIndex = kwargs.startIndex ?? 0;
      const shape = getExactlyOneTensor(inputs).shape;
      const featureLength = shape[shape.length - 1];
      const sequenceLength = shape[shape.length - 2];
      // trim to match the length of the input sequence, which might be less
      // than the sequence_length of the layer.
      const positionEmbeddings = this.positionEmbeddings.read().slice(
        [kwargs.startIndex, 0], [sequenceLength, featureLength]);
      return positionEmbeddings.broadcastTo(shape);
    });
  }

  override computeOutputShape(inputShape: Shape): Shape {
    return inputShape;
  }
}
serialization.registerClass(PositionEmbedding);
