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
 *  Tokenizer layers.
 */

/* Original source: keras-nlp/tokenizer.py */
import { Tensor1D } from '@tensorflow/tfjs-core';

import { Layer } from '../../engine/topology';
import { NotImplementedError, ValueError } from '../../errors';

export declare interface TokenizerOptions {
  mode?: 'tokenize' | 'detokenize';
}

/**
 * Base class for Tokenizers.
 *
 *  Tokenizers in the tfjs library should all subclass this layer.
 *  The class provides two core methods `tokenize()` and `detokenize()` for
 *  going from plain text to sequences and back. A tokenizer is a subclass of
 *  `Layer` and can be combined with other layers in a `tf.sequential` model.
 *
 *  Subclassers should always implement the `tokenize()` method, which will also
 *  be the default when calling the layer directly on inputs.
 *
 *  Subclassers can optionally implement the `detokenize()` method if the
 *  tokenization is reversible. Otherwise, this can be skipped.
 *
 *  Subclassers should implement `get_vocabulary()`, `vocabulary_size()`,
 *  `token_to_id()` and `id_to_token()` if applicable. For some simple
 *  "vocab free" tokenizers, such as a whitespace splitter shown below, these
 *  methods do not apply and can be skipped.
 *
 *  Example:
 *
 *  ```js
 *  class WhitespaceSplitterTokenizer extends Tokenizer {
 *    tokenize(inputs: Tensor1D): Tensor1D[] {
 *      const stringInputs = inputs.dataSync() as unknown as string[];
 *      return stringInputs.map(input => tensor1d(input.split(' ')));
 *    }
 *
 *    override detokenize(inputs: Tensor1D[]): Tensor1D {
 *      const stringInputs = inputs.map(
 *        input => input.dataSync() as unknown as string[]);
 *      return tensor1d(stringInputs.map(str => str.join(' ')));
 *    }
 *  }
 *
 * const tokenizer = new WhitespaceSplitterTokenizer();
 *
 * tokenizer.tokenize(tensor1d(['this is a test']));
 *
 * tokenizer.detokenize([tensor1d(['this', 'is', 'a', 'test'])]);
 * ```
 */
export abstract class Tokenizer extends Layer {
  /**
   * Transform input tensors of strings into output tokens.
   *
   * @param inputs Input tensor.
   * @param kwargs Additional keyword arguments.
   */
  abstract tokenize(inputs: Tensor1D): Tensor1D[];

  /**
   * Transform tokens back into strings.
   *
   * @param inputs Input tensor.
   * @param kwargs Additional keyword arguments.
   */
  detokenize(inputs: Tensor1D[]): Tensor1D {
    throw new NotImplementedError(
      `No implementation of 'detokenize()' was found for
      ${this.constructor.name}.`
    );
  }

  /**
   * Get the tokenizer vocabulary as a list of strings terms.
   */
  get vocabulary(): string[] {
    throw new NotImplementedError(
      `No implementation of 'vocabulary()' was found for
      ${this.constructor.name}.`
    );
  }

  /**
   * Returns the total size of the token id space.
   */
  get vocabularySize(): number {
    throw new NotImplementedError(
      `No implementation of 'vocabularySize()' was found for
      ${this.constructor.name}.`
    );
  }

  /**
   * Convert an integer id to a string token.
   */
  idToToken(id: number): string {
    throw new NotImplementedError(
      `No implementation of 'idToToken()' was found for
      ${this.constructor.name}.`
    );
  }

  /**
   * Convert an integer id to a string token.
   */
  tokenToId(token: string): number {
    throw new NotImplementedError(
      `No implementation of 'tokenToId()' was found for
      ${this.constructor.name}.`
    );
  }

  override call(
    inputs: Tensor1D|Tensor1D[],
    {mode = 'tokenize'}: TokenizerOptions={}
  ): Tensor1D|Tensor1D[] {

    if (mode === 'tokenize') {
      if (inputs instanceof Array) {
        throw new ValueError(`tokenize expects Tensor1D, not Tensor1D[].`);
      }
      return this.tokenize(inputs);
    }

    if (mode === 'detokenize') {
      if (!(inputs instanceof Array)) {
        throw new ValueError(`detokenize expects Tensor1D[], not Tensor1D.`);
      }
      return this.detokenize(inputs);
    }

    throw new ValueError(`Input mode=${mode} is not supported.`);
  }
}
