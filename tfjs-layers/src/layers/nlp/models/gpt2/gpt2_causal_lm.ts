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
 * GPT2 Causal LM (Language Model).
 */

/* Original source: keras-nlp/models/gpt2/gpt2_causal_lm.py */
import { NamedTensorMap, Tensor, serialization } from '@tensorflow/tfjs-core';

import { GPT2Preprocessor } from './gpt2_preprocessor';
import { NotImplementedError } from '../../../../errors';
import { Layer } from '../../../../exports_layers';
import { LayerArgs } from '../../../../engine/topology';
import { Embedding } from '../../../../layers/embeddings';
import { Shape } from '../../../../keras_format/common';
import { GenerativeTask } from '../generative_task';
import { GPT2Backbone } from './gpt2_backbone';
import { PipelineModelArgs } from '../../utils';
import { Kwargs } from '../../../../types';

declare interface ReverseEmbeddingArgs extends LayerArgs {
  embedding: Embedding;
}

class ReverseEmbedding extends Layer {
  protected embedding: Embedding;

  constructor(args: ReverseEmbeddingArgs) {
    super(args);
    this.embedding = args.embedding;
  }

  override call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    throw new NotImplementedError();
  }

  override computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    throw new NotImplementedError();
  }

}

export declare interface GPT2CausalLMArgs extends PipelineModelArgs {
  /**
   * A `GPT2Backbone` instance.
   */
  backbone: GPT2Backbone;

  /**
   * Optional `GPT2CausalLMPreprocessor`.
   * If `null`, this model will not apply preprocessing, and inputs should be
   * preprocessed before calling the model.
   */
  preprocessor?: GPT2Preprocessor;
}

/**
 * An end-to-end GPT2 model for causal langauge modeling.
 *
 * A causal language model (LM) predicts the next token based on previous
 * tokens. This task setup can be used to train the model unsupervised on
 * plain text input, or to autoregressively generate plain text similar to
 * the data used for training. This task can be used for pre-training or
 * fine-tuning a GPT-2 model, simply by calling `fit()`.
 *
 * This model has a `generate()` method, which generates text based on a
 * prompt. The generation strategy used is controlled by an additional
 * sampler` argument on `compile()`.
 * By default, the top k results will be returned.
 *
 * This model can optionally be configured with a `preprocessor` layer, in
 * which case it will automatically apply preprocessing to string inputs during
 * fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
 * when creating the model with `fromPreset()`.
 *
 * Disclaimer: Pre-trained models are provided on an "as is" basis, without
 * warranties or conditions of any kind. The underlying model is provided by a
 * third party and subject to a separate license, available
 * here](https://github.com/openai/gpt-2).
 *
 * Use `generate()` to do text generation.
 * ```js
 * const gpt2LM = GPT2CausalLM.fromPreset('gpt2_base_en');
 * gpt2LM.generate("I want to say", max_length=30);
 * // Generate with batched prompts.
 * gpt2LM.generate(["This is a", "Where are you"], max_length=30);
 * ```
 *
 * Use `generate()` without preprocessing.
 * ```js
 * // Prompt the model with `5338, 318` (the token ids for `"Who is"`).
 * // Use `"paddingMask"` to indicate values that should not be overridden.
 * const prompt = {
 *  tokenIds: tf.tensor([[5338, 318, 0, 0, 0], [5338, 318, 0, 0, 0]]),
 *  paddingMask: tf.tensor([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]]),
 * };
 * const gpt2LM = GPT2CausalLM.from_preset('gpt2_base_en', null);
 * gpt2LM.generate(prompt);
 * ```
 *
 * Call `fit()` on a single batch.
 * ```js
 * const features = ['The quick brown fox jumped.', 'I forgot my homework.'];
 * const gpt2LM = GPT2CausalLM.from_preset('gpt2_base_en');
 * gpt2LM.fit(features, {batchSize: 2});
 * ```
 *
 * Call `fit()` without preprocessing.
 * ```js
 * const x = {
 *  tokenIds: tf.tensor([[50256, 1, 2, 3, 4], [50256, 1, 2, 3, 4]]),
 *  paddingMask: tf.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
 * };
 * const y = tf.tensor([[1, 2, 3, 4, 50256], [1, 2, 3, 4, 50256]]);
 * const sw = tf.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]);
 * const gpt2LM = GPT2CausalLM.from_preset('gpt2_base_en', null);
 * gpt2LM.fit(x, y, {sampleWeight: sw, batchSize: 2});
 * ```
 *
 * Custom backbone and vocabulary.
 * ```js
 * const features = ["a quick fox.", "a fox quick."];
 * const vocab = {"<|endoftext|>": 0, "a": 4, "Ġquick": 5, "Ġfox": 6};
 * const merges = [
 *  "Ġ q", "u i", "c k", "ui ck", "Ġq uick", "Ġ f", "o x", "Ġf ox"
 * ];
 * const tokenizer = new GPT2Tokenizer({vocabulary: vocab, merges});
 * const preprocessor =  new GPT2CausalLMPreprocessor({
 *  tokenizer,
 *  sequence_length: 128,
 * });
 * const backbone = new GPT2Backbone({
 *  vocabularysize: 30552,
 *  numlayers: 4,
 *  numheads: 4,
 *  hiddendim: 256,
 *  intermediatedim: 512,
 *  maxSequenceLength: 128,
 * });
 * const gpt2LM = new GPT2CausalLM({backbone, preprocessor});
 * gpt2LM.fit(features, {batch_size: 2});
 * ```
 */
export class GPT2CausalLM extends GenerativeTask {
  /** @nocollapse */
  static override className = 'GPT2CausalLM';

  constructor(args: GPT2CausalLMArgs) {
    super(args);
    throw new NotImplementedError(`Uses ${ReverseEmbedding}.`);
  }

  static override presets<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>
  ): {} {
    throw new NotImplementedError();
  }

  /**
   * Forward pass of `GPT2CausalLM` with cache.
   *
   * `callWithCache` adds an additional forward pass for the model for
   * autoregressive inference. Unlike calling the model directly, this method
   * allows caching previous key/value Tensors in multi-head attention layer,
   * and avoids recomputing the outputs of seen tokens.
   *
   * @param tokenIds a dense int Tensor with shape `[batchSize, maxLength]`.
   * @param cache a dense float Tensor, the cache of key and value.
   * @param cacheUpdateIndex Integer. The index of current inputs in the whole
   *  sequence.
   * @returns [logits, hiddenStates, cache], where `logits` is the
   *  language model logits for the input tokenIds, `hiddenStates` is
   *  the final hidden representation of the input tokens, and `cache` is
   *  the decoding cache.
   */
  callWithCache(
    tokenIds: Tensor,
    cache: Tensor,
    cacheUpdateIndex: number
  ): [Tensor, Tensor, Tensor] {
    throw new NotImplementedError();
  }

  /**
   * Build an empty cache for use with `callWithCache()`.
   */
  private buildCache(tokenIds: Tensor): [Tensor, Tensor] {
    throw new NotImplementedError();
  }

  /**
   * A compilable generation function for a single batch of inputs.
   *
   * This function represents the inner generation function for a single batch
   *  of inputs.
   *
   * @param inputs An object with two keys `tokenIds` and `paddingMask` and
   *  batched tensor values.
   * @param endTokenId The id of the end token to stop on. If all
   *  sequences have produced a new `endTokenId`, generation will stop.
   */
  override generateStep(
    inputs: NamedTensorMap,
    endTokenId: number
  ): NamedTensorMap {
    throw new NotImplementedError(`Uses ${this.buildCache}`);
  }
}
serialization.registerClass(GPT2CausalLM);
