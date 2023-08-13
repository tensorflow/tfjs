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
import { AdamOptimizer, Tensor, logicalAnd, onesLike, scalar, serialization, softmax, stack, tensor, topk, zeros, zerosLike } from '@tensorflow/tfjs-core';

import { NotImplementedError } from '../../../../errors';
import { Layer } from '../../../../exports_layers';
import { LayerArgs, SymbolicTensor } from '../../../../engine/topology';
import { Embedding } from '../../../../layers/embeddings';
import { Shape } from '../../../../keras_format/common';

import { GPT2TensorMap, GenerativeTask } from '../generative_task';
import { sliceUpdate } from '../../utils';
import { GPT2Backbone } from './gpt2_backbone';
import { sparseCategoricalCrossentropy } from '../../../../losses';
import { Kwargs } from '../../../../types';
import { TransformerDecoder } from '../../modeling/transformer_decoder';
import { GPT2CausalLMPreprocessor } from './gpt2_causal_lm_preprocessor';

declare interface ReverseEmbeddingArgs extends LayerArgs {
  embedding: Embedding;
}

class ReverseEmbedding extends Layer {
  protected embedding: Embedding;

  constructor(args: ReverseEmbeddingArgs) {
    super(args);
    this.embedding = args.embedding;
  }

  override call(inputs: Tensor, kwargs: Kwargs): Tensor|Tensor[] {
    const kernel = this.embedding.embeddings.read().transpose();
    return inputs.matMul(kernel);
  }

  override computeOutputShape(inputShape: Shape): Shape|Shape[] {
    return [inputShape[0], this.embedding.embeddings.shape[0]];
  }

}

export declare interface GPT2CausalLMArgs {
  /**
   * A `GPT2Backbone` instance.
   */
  backbone: GPT2Backbone;

  /**
   * Optional `GPT2CausalLMPreprocessor`.
   * If `null`, this model will not apply preprocessing, and inputs should be
   * preprocessed before calling the model.
   */
  preprocessor?: GPT2CausalLMPreprocessor;
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

  constructor(args: GPT2CausalLMArgs) {
    const inputs = args.backbone.input;
    const x = args.backbone.apply(inputs) as SymbolicTensor;
    // Use token embedding weights to project from the token representation
    // to vocabulary logits.
    const outputs = new ReverseEmbedding({
      embedding: args.backbone.tokenEmbedding,
      name: 'reverse_embedding',
    }).apply(x) as SymbolicTensor;

    // Instantiate using Functional API Model constructor.
    super({
      inputs,
      outputs,
      includePreprocessing: args.preprocessor != null,
      ...args,
    });
    this.backbone = args.backbone;
    this.preprocessor = args.preprocessor;

    // Default complation.
    this.compile({
      loss: (yTrue: Tensor, yPred: Tensor) =>
        sparseCategoricalCrossentropy(yTrue, yPred, true),
      optimizer: new AdamOptimizer(2e-5, 0.9, 0.999),
      metrics: ['sparseCategoricalCrossentropy'],
    });
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
    const tokenEmbedding = this.backbone.getLayer('token_embedding')
      .apply(tokenIds) as Tensor;
    const positionEmbedding = this.backbone.getLayer('position_embedding')
      .apply(tokenEmbedding, {startIndex: cacheUpdateIndex}) as Tensor;
    let x = this.backbone.getLayer('embeddings_add')
      .apply([tokenEmbedding, positionEmbedding]) as Tensor;
    x = this.backbone.getLayer('embeddings_dropout')
      .apply(x) as Tensor;

    // Each decoder layer has a cache; we update them separately.
    const caches = [];
    let currentCache: Tensor;
    let nextCache: Tensor;
    for (let i = 0; i < (this.backbone as GPT2Backbone).numLayers; i++) {
      currentCache = cache.gather([0], 1).squeeze();
      [x, nextCache] = (
        this.backbone.getLayer(`transformer_layer_${i}`) as TransformerDecoder
      ).callAndReturnCaches(
        x,
        {
          selfAttentionCache: currentCache,
          selfAttentionCacheUpdateIndex: cacheUpdateIndex
        }
      );
      caches.push(nextCache);
    }
    cache = stack(caches, 1);
    x = this.backbone.getLayer('layer_norm').apply(x) as Tensor;
    const hiddenStates = x;
    const logits = this.getLayer('reverse_embedding').apply(x) as Tensor;
    return [logits, hiddenStates, cache];
  }

  /**
   * Build an empty cache for use with `callWithCache()`.
   */
  private buildCache(tokenIds: Tensor): [Tensor, Tensor] {
    const batchSize = tokenIds.shape[0];
    const maxLength = tokenIds.shape[1];
    const numLayers = (this.backbone as GPT2Backbone).numLayers;
    const numHeads = (this.backbone as GPT2Backbone).numHeads;
    const headDim = (this.backbone as GPT2Backbone).hiddenDim / numHeads;
    const shape = [batchSize, numLayers, 2, maxLength, numHeads, headDim];
    let cache = zeros(shape);
    // Seed the cache.
    const cacheCall = this.callWithCache(tokenIds, cache, 0);
    const hiddenStates = cacheCall[1];
    cache = cacheCall[2];
    return [hiddenStates, cache];
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
    inputs: GPT2TensorMap,
    endTokenId?: number
  ): GPT2TensorMap {
    let tokenIds = inputs.tokenIds;
    let paddingMask = inputs.paddingMask;
    // Create and seed cache with a single forward pass.
    const [hiddenStates, cache] = this.buildCache(tokenIds);
    // Compute the lengths of all user inputted tokens ids.
    const rowLengths = paddingMask.cast('int32').sum(-1);
    // Start at the first index that has no user inputted id.
    const index = rowLengths.min().arraySync() as number;

    const self = this;
    function next(
      prompt: Tensor,
      cache: Tensor,
      index: number
    ): [Tensor, Tensor, Tensor] {
      // The cache index is the index of our previous token.
      const cacheUpdateIndex = index - 1;
      const batchSize = prompt.shape[0];
      prompt = prompt.slice([0, cacheUpdateIndex], [batchSize, 1]);
      let logits: Tensor;
      let hiddenStates: Tensor;
      [logits, hiddenStates, cache] =
        self.callWithCache(prompt, cache, cacheUpdateIndex);
      return [
        logits.squeeze([1]),
        hiddenStates.squeeze([1]),
        cache
      ];
    }
    tokenIds =
      this.sampler(next, tokenIds, cache, index, paddingMask, endTokenId, hiddenStates);

    // Compute an output padding mask with the token ids we updated.
    if (endTokenId != null) {
      // Build a mask of `endTokenId` locations not in the original
      // prompt (not in locations where `paddingMask` is True).
      let endLocations = logicalAnd(
        tokenIds.equal(endTokenId),
        paddingMask.logicalNot(),
      );
      endLocations = endLocations.cast('int32');
      // Use cumsum to get ones in all locations after end_locations.
      const cumsum = endLocations.cumsum(-1).cast('int32');
      const overflow = cumsum.sub(endLocations);
      // Our padding mask is the inverse of these overflow locations.
      paddingMask = overflow.cast('bool').logicalNot();
    } else {
      // Without early stopping, all locations will have been updated.
      paddingMask = onesLike(tokenIds).cast('bool');
    }
    return {tokenIds, paddingMask};
  }

  private sampler(
    next: (prompt: Tensor, cache: Tensor, index: number)
      => [Tensor, Tensor, Tensor],
    prompt: Tensor,
    cache?: Tensor,
    index?: number,
    mask?: Tensor,
    endTokenId?: number,
    hiddenStates?: Tensor
  ): Tensor {
    const temperature = 1.0;
    const maxLength = prompt.shape[prompt.shape.length - 1];
    index = index ?? 0;
    mask = mask == null ? zerosLike(prompt).cast('bool') : mask.cast('bool');
    cache = cache ?? tensor([]);

    function cond(prompt: Tensor): boolean {
      if (endTokenId == null) {
        return true;
      }
      // Stop if all sequences have produced a *new* endTokenId.
      const endTokens = prompt.equal(endTokenId).logicalAnd(mask.logicalNot());
      const promptDone = endTokens.any(-1);
      return promptDone.all().logicalNot().arraySync() as number === 1;
    }

    function generateNextToken(probabilities: Tensor): Tensor {
      // Filter out top-k tokens.
      const k = 5;
      const {values, indices} = topk(probabilities, k, false);

      function randomSample(values: Tensor, probabilities: Tensor): number {
        const probabilitiesArr = probabilities.arraySync() as number[];
        let sample = Math.random() * probabilitiesArr.reduce((a, b) => a + b, 0);
        const value = (values.arraySync() as number[]).find((val, index) => {
          return (sample -= probabilitiesArr[index]) <= 0;
        });
        return value;
      }
      // Sample the indices with the distribution.
      const nextToken = randomSample(indices, values);
      const nextTokenTensor = scalar(nextToken, 'int32');
      return nextTokenTensor;
    }

    let iter = 0;
    let logits: Tensor;
    while (iter < maxLength - index && cond(prompt)) {
      // Compute the softmax distribution for the next token.
      [logits, hiddenStates, cache] = next(prompt, cache, index);
      let probabilities = softmax(logits.div(temperature));
      // Compute next token.
      let nextToken = generateNextToken(probabilities);
      // Don't overwrite anywhere mask is True.
      nextToken = nextToken.cast(prompt.dtype);
      nextToken = nextToken.where(
        mask.gather([index], 1).squeeze(),
        prompt.gather([index], 1).squeeze()
      );
      // Update the prompt with the next token.
      nextToken = nextToken.expandDims(-1);
      prompt = sliceUpdate(prompt, [0, index], nextToken);

      index += 1;
      iter += 1;
    }
    return prompt;
  }
}
