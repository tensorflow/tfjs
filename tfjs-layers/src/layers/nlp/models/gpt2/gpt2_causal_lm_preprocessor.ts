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
 * GPT2 Causal LM preprocessor layer.
 */

/* Original source: keras-nlp/models/gpt2/gpt2_preprocessor.py */
import { Tensor, serialization } from '@tensorflow/tfjs-core';

import { GPT2Preprocessor, PreprocessorOutputs } from './gpt2_preprocessor';
import { NotImplementedError } from 'tfjs-layers/src/errors';

function packXYSampleWeight(
  x: PreprocessorOutputs, y?: Tensor, sampleWeight?: Tensor):
  PreprocessorOutputs
  | [PreprocessorOutputs, Tensor]
  | [PreprocessorOutputs, Tensor, Tensor] {

  if (y === undefined) {
    return x;
  } else if (sampleWeight === undefined) {
    return [x, y];
  } else {
    return [x, y, sampleWeight];
  }
}

export declare interface GPT2PreprocessorOptions {
  /**
   * Label data. Should always be `null` as the layer generates labels.
   */
  y?: Tensor;

  /**
   * Label weights. Should always be `null` as the layer generates label
   *  weights.
   */
  sampleWeight?: Tensor;

  /**
   * Pass to override the configured `sequenceLength` of the layer.
   */
  sequenceLength?: number;
}

/**
  * GPT2 Causal LM preprocessor.
  *
  * This preprocessing layer is meant for use with
  * `GPT2CausalLM`. By default, it will take in batches of
  * strings, and return outputs in a `[x, y, sampleWeight]` format, where the
  * `y` label is the next token id in the `x` sequence.
  *
  * For use with generation, the layer also exposes two methods
  * generatePreprocess()` and `generatePostprocess()`. When this preprocessor
  * is attached to a `GPT2CausalLM` instance, these methods
  * will be called implicitly in `generate()`. They can also be called
  * standalone (e.g. to precompute preprocessing inputs for generation in a
  * separate process).
  *
  * Examples:
  * ```js
  * // Load the preprocessor from a preset.
  * const preprocessor = GPT2CausalLMPreprocessor.from_preset('gpt2_base_en');
  *
  * // Tokenize and pack a single sentence.
  * const sentence = tf.scalar('League of legends');
  * preprocessor.apply(sentence);
  * // Same output.
  * preprocessor('League of legends');
  *
  * // Tokenize a batch of sentences.
  * const sentences = tf.constant(['Taco tuesday', 'Fish taco please!']);
  * preprocessor.apply(sentences);
  * // Same output.
  * preprocessor.apply(['Taco tuesday', 'Fish taco please!']);
  * ```
 */
export class GPT2CausalLMPreprocessor extends GPT2Preprocessor {

  override getConfig(): serialization.ConfigDict {
    const config = {
      sequenceLength: this.sequenceLength,
      addStartToken: this.addStartToken,
      addEndToken: this.addEndToken,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  override call(
    inputs: Tensor|Tensor[],
    kwargs: GPT2PreprocessorOptions
  ): Tensor|Tensor[] {
    const output = this.callAndPackArgs(inputs, kwargs);
    if (kwargs.y) {
      return (output as [PreprocessorOutputs, Tensor])[0].tokenIds;
    }
    return (output as PreprocessorOutputs).tokenIds;
  }

  /**
   * Calls the layer and returns extra information like the paddingMask used to
   * pack the sequence, the label data, and the sample weights used.
   */
  override callAndPackArgs(inputs: Tensor|Tensor[], kwargs: GPT2PreprocessorOptions):
    PreprocessorOutputs
    | [PreprocessorOutputs, Tensor]
    | [PreprocessorOutputs, Tensor, Tensor] {

    throw new NotImplementedError(`Uses ${packXYSampleWeight}`);
  }

  /**
   * Covert strings to integer token input for generation.
   *
   * Similar to calling the layer for training, this method takes in strings
   * or tensor strings, tokenizes and packs the input, and computes a padding
   * mask masking all inputs not filled in with a padded value.
   *
   * Unlike calling the the layer for training, this method does not compute
   * labels and will never append a `tokenizer.endTokenId` to the end of
   * the sequence (as generation is expected to continue at the end of the
   * inputted prompt).
   */
  generatePreprocess(x: Tensor, sequenceLength?: number): PreprocessorOutputs {
    throw new NotImplementedError();
  }

  /**
   * Covert integer token output to strings for generation.
   *
   * This method reverses `generatePreprocess()`, by first removing all
   * padding and start/end tokens, and then converting the integer sequence
   * back to a string.
   */
  generatePostprocess(x: Tensor): PreprocessorOutputs {
    throw new NotImplementedError();
  }

}
serialization.registerClass(GPT2Preprocessor);
