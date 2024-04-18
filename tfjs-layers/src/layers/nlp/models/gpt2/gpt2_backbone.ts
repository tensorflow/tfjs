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
 *  Base class for Backbone models.
 */

/* Original source: keras_nlp/models/gpt2/gpt2_backbone.py */
import { serialization } from '@tensorflow/tfjs-core';

import { RandomNormal } from '../../../../initializers';
import { input } from '../../../../exports';
import { Embedding } from '../../../embeddings';
import { SymbolicTensor } from '../../../../engine/topology';
import { PositionEmbedding } from '../../modeling/position_embedding';
import { add } from '../../../../exports_layers';
import { Dropout } from '../../../core';
import { TransformerDecoder } from '../../modeling/transformer_decoder';
import { getActivation } from '../../../../activations';
import { LayerNormalization } from '../../../normalization';
import { Backbone } from '../backbone';

function gpt2KernelInitializer(stddev = 0.02) {
  return new RandomNormal({stddev});
}

export interface GPT2BackboneArgs  {
  /**
   * Integer. The size of the token vocabulary.
   */
  vocabularySize: number;

  /**
   * Integer. The number of transformer layers.
   */
  numLayers: number;

  /**
   * Integer. The number of attention heads for each transformer.
   * The hidden size must be divisible by the number of attention heads.
   */
  numHeads: number;

  /**
   * Integer. The size of the transformer encoding and pooler layers.
   */
  hiddenDim: number;

  /**
   * Integer. The output dimension of the first Dense layer in a two-layer
   * feedforward network for each transformer.
   */
  intermediateDim: number;

  /**
   * Float. Dropout probability for the Transformer encoder.
   * Defaults to 0.2.
   */
  dropout?: number;

  /**
   * Integer. The maximum sequence length that this encoder can consume.
   * If `null`, `maxSequenceLength` uses the value from sequence length.
   * This determines the variable shape for positional embeddings.
   * Defaults to 1024.
   */
  maxSequenceLength?: number;
}

/**
 * GPT-2 core network with hyperparameters.
 *
 * This network implements a Transformer-based decoder network,
 * Generative Pretrained Transformer-2 (GPT-2), as described in
 * ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
 * It includes the embedding lookups and transformer layers.
 *
 * The default constructor gives a fully customizable, randomly initialized
 * GPT-2 model with any number of layers, heads, and embedding
 * dimensions. To load preset architectures and weights, use the `fromPreset`
 * constructor.
 *
 * Disclaimer: Pre-trained models are provided on an "as is" basis, without
 * warranties or conditions of any kind. The underlying model is provided by a
 * third party and subject to a separate license, available
 * [here](https://github.com/openai/gpt-2).
 *
 *
 * Example usage:
 * ```js
 * const tokenIds = tf.ones([1, 12]), dtype="int32");
 * const paddingMask = tf.tensor(
 *  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], 'int32');
 *
 * # Pretrained GPT-2 decoder.
 * model = GPT2Backbone.fromPreset("gpt2_base_en");
 * model.apply(inputData, {paddingMask});
 *
 * # Randomly initialized GPT-2 decoder with custom config.
 * model = kerasNlp.models.GPT2Backbone({
 *     vocabularySize: 50257,
 *     numLayers: 12,
 *     numHeads: 12,
 *     hiddenDim: 768,
 *     intermediateDim: 3072,
 *     maxSequenceLength: 1024,
 * });
 * model.apply(inputData, {paddingMask});
 * ```
 */
export class GPT2Backbone extends Backbone {
  /** @nocollapse */
  static override className = 'GPT2Backbone';

  private vocabularySize: number;
  private numLayers: number;
  private numHeads: number;
  private hiddenDim: number;
  private intermediateDim: number;
  private dropout: number;
  private maxSequenceLength: number;

  constructor(args: GPT2BackboneArgs) {
    args.dropout = args.dropout ?? 0.1;
    args.maxSequenceLength = args.maxSequenceLength ?? 1024;

    // Inputs
    const tokenIds = input({shape: [null], dtype: 'int32', name: 'token_ids'});
    const paddingMask =
      input({shape: [null], dtype: 'int32', name: 'padding_mask'});

    // Embed tokens, positions.
    const tokenEmbedding = new Embedding({
      inputDim: args.vocabularySize,
      outputDim: args.hiddenDim,
      embeddingsInitializer: gpt2KernelInitializer(0.01),
      name: 'token_embedding',
    }).apply(tokenIds) as SymbolicTensor;

    const positionEmbedding = new PositionEmbedding({
      initializer: gpt2KernelInitializer(0.02),
      sequenceLength: args.maxSequenceLength,
      name: 'position_embedding',
    }).apply(tokenEmbedding) as SymbolicTensor;

    // Sum and apply dropout to embeddings.
    let x = add({name: 'embeddings_add'})
      .apply([tokenEmbedding, positionEmbedding]) as SymbolicTensor;
    x = new Dropout({rate: args.dropout, name: 'embeddings_dropout'})
      .apply(x) as SymbolicTensor;

    // Apply successive transformer decoder blocks.
    for(let i = 0; i < args.numLayers; i++) {
      x = new TransformerDecoder({
        intermediateDim: args.intermediateDim,
        numHeads: args.numHeads,
        dropout: args.dropout,
        layerNormEpsilon: 1e-05,
        activation: getActivation('gelu'),
        kernelInitializer: gpt2KernelInitializer(0.02),
        normalizeFirst: true,
        name: `transformer_layer_${i}`,
      }).apply(x, {decoderPaddingMask: paddingMask}) as SymbolicTensor;
    }

    const sequenceOutput = new LayerNormalization({
      name: 'layer_norm',
      axis: -1,
      epsilon: 1e-05,
      dtype: 'float32',
    }).apply(x) as SymbolicTensor;

    // Instantiate using Functional API Model constructor.
    super({
      inputs: [tokenIds, paddingMask],
      outputs: sequenceOutput,
      name: 'gpt2_backbone'
    });
    this.vocabularySize = args.vocabularySize;
    this.numLayers = args.numLayers;
    this.numHeads = args.numHeads;
    this.hiddenDim = args.hiddenDim;
    this.intermediateDim = args.intermediateDim;
    this.dropout = args.dropout ?? 0.1;
    this.maxSequenceLength = args.maxSequenceLength ?? 1024;
  }

  override getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      vocabularySize: this.vocabularySize,
      numLayers: this.numLayers,
      numHeads: this.numHeads,
      hiddenDim: this.hiddenDim,
      intermediateDim: this.intermediateDim,
      dropout: this.dropout,
      maxSequenceLength: this.maxSequenceLength,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  override get tokenEmbedding(): Embedding {
    return this.getLayer('token_embedding') as Embedding;
  }
}
serialization.registerClass(GPT2Backbone);
