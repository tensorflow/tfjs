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
 * Sampler classes.
 */

/* Original source: keras_nlp/samplers/sampler.py */
import { Tensor, serialization, softmax, tensor, tidy, topk, where, zerosLike } from '@tensorflow/tfjs-core';
import { sliceUpdate } from './utils';

export type NextFn =
  (prompt: Tensor, cache: Tensor, index: number) => [Tensor, Tensor, Tensor];

export interface SamplerArgs {
  /**
   * Float.
   * Used to control the randomness of the sampling.
   * The higher the temperature, the more diverse the samples.
   * Defaults to `1.0`.
   */
  temperature?: number;
}

/**
 * Base sampler class.
 *
 * This base class can be extended to implement different auto-regressive
 * sampling methods. Subclasses can either:
 *
 * - Override the `getNextToken()` method, which computes the next token
 * based on a probability distribution over all possible vocab entries.
 *
 * - Override `apply()`, if the sampling method needs additional information
 * beyond the next tokens probability distribution to sample a sequence.
 * Please check available subclass samplers for examples.
 *
 * Examples:
 * // TODO(pforderique): Add examples.
 */
export abstract class Sampler {
  temperature: number;

  constructor(args: SamplerArgs) {
    this.temperature = args.temperature ?? 1.0;
  }

  /**
   * @param next A function which takes in the
   *     `prompt, cache, index` of the current generation loop, and outputs
   *     a tuple `(logits, hiddenStates, cache)` with `logits` being the
   *     logits of next token, `hiddenStates` being the representation of
   *     the next token, and `cache` for next iteration.
   * @param prompt A 2D integer tensor with shape `[batchSize, maxLength]`. This
   *     tensor will be iteratively updated column by column with new sampled
   *     values, starting at `index`.
   * @param cache A tensor or nested structure of tensors that will be
   *     updated by each call to `next`. This can be used to cache
   *     computations from early iterations of the generative loop.
   * @param index The first index of `prompt` to start sampling at.
   *     Usually this is set as the length of the shortest non-padded
   *     sequence in `prompt`.
   * @param mask A 2D integer tensor with the same shape as `prompt`.
   *     Locations which are `true` in the mask are never updated during
   *     sampling. Usually used to mark all locations in the dense prompt
   *     tensor which were present in a user input.
   * @param endTokenId The token marking the end of the sequence. If
   *     specified, sampling will stop as soon as all sequences in the prompt
   *     produce a `endTokenId` in a location where `mask` is `false`.
   */
  apply(
    next: NextFn,
    prompt: Tensor,
    cache: Tensor = tensor([]),
    index: number = 0,
    mask?: Tensor,
    endTokenId?: number,
    hiddenStates?: Tensor
  ): Tensor {
    const maxLength = prompt.shape[prompt.shape.length - 1];
    mask = mask == null ? zerosLike(prompt).cast('bool') : mask.cast('bool');

    function cond(prompt: Tensor, index: number): boolean {
      if (index >= maxLength) {
        return false;
      }
      if (endTokenId == null) {
        return true;
      }
      // Stop if all sequences have produced a *new* endTokenId.
      const endTokens = prompt.equal(endTokenId).logicalAnd(mask.logicalNot());
      const promptDone = endTokens.any(-1).cast('bool');
      return promptDone.all().logicalNot().arraySync() as number === 1;
    }

    let iter = 0;
    let logits: Tensor;
    const maxIterations = maxLength - index;
    while (iter <= maxIterations && cond(prompt, index)) {
      tidy(() => {
        // Compute the softmax distribution for the next token.
        const oldCache = cache;
        [logits, hiddenStates, cache] = next(prompt, cache, index);
        oldCache.dispose();

        const probabilities = softmax(logits.div(this.temperature));
        // Compute next token.
        let nextToken = this.getNextToken(probabilities);
        // Don't overwrite anywhere mask is True.
        nextToken = nextToken.cast(prompt.dtype);
        nextToken = where(
          mask.gather(index, 1),
          prompt.gather(index, 1),
          nextToken,
        );
        // Update the prompt with the next token.
        nextToken = nextToken.expandDims(-1);
        const oldPrompt = prompt;
        prompt = sliceUpdate(prompt, [0, index], nextToken);
        oldPrompt.dispose();

        index += 1;
        iter += 1;

        // Keep these tensors.
        return {prompt, cache};
      });
    }
    return prompt;
  }

  /**
   * Get the next token based on given probability distribution over tokens.
   * Subclasses must implement this method.
   *
   * @param probabilities the probability distribution for next
   *    token over all vocab tokens.
   */
  abstract getNextToken(probabilities: Tensor): Tensor;

  static fromConfig<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>,
    config: serialization.ConfigDict): T {

    return new cls(config);
  }

  getConfig(): serialization.ConfigDict {
    return {temperature: this.temperature};
  }
}

export interface TopKSamplerArgs extends SamplerArgs {
  /**
   * Integer, the `k` value of top-k.
   * Defaults to 5.
   */
  k?: number;

  /**
   * Integer, the random seed.
   * Defaults to `null`,
   */
  seed?: number;
}

/**
 * Top-K Sampler class.
 *
 * This sampler implements top-k search algorithm. Briefly, top-k algorithm
 * randomly selects a token from the tokens of top K probability, with
 * selection chance determined by the probability.
 *
 * Examples:
 * // TODO(pforderique): Add examples.
 */
export class TopKSampler extends Sampler {
  k: number;
  seed: number;

  constructor(args: TopKSamplerArgs) {
    super(args);
    this.k = args.k ?? 5;
    this.seed = args.seed;
  }

  override getNextToken(probabilities: Tensor): Tensor {
    // Filter out top-k tokens.
    const {values, indices} = topk(probabilities, this.k, false);

    // TODO(mattSoulanille): Investigate whether this should use binsearch.
    function randomSample(probabilities: Tensor): Tensor {
      const probabilitiesArr = probabilities.arraySync() as number[][];
      const samplesArr = [];

      for (const probDistribution of probabilitiesArr) {
        let sample =
          Math.random() * probDistribution.reduce((a, b) => a + b, 0);
        const sampleIdx = probDistribution.findIndex(val => {
          return (sample -= val) <= 0;
        });
        samplesArr.push(sampleIdx);
      }
      return tensor(samplesArr, null, 'int32');
    }
    // Sample the nextToken from the probability distribution.
    const nextToken = randomSample(values);
    return indices.gather(nextToken, 1, 1);
  }

  override getConfig(): serialization.ConfigDict {
    const config = {
      k: this.k,
      seed: this.seed,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
