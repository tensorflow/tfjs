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
 * GPT-2 tokenizer layer.
 */

/* Original source: keras-nlp/models/gpt2/gpt2_tokenizer.py */
import { serialization } from '@tensorflow/tfjs-core';

import { LayerArgs } from '../../../../engine/topology';
import { BytePairTokenizer } from '../../tokenizers';
import { ValueError } from '../../../../errors';

export declare interface GPT2TokenizerArgs extends LayerArgs {
  /**
   * Maps token to integer ids
   */
  vocabulary: Map<string, number>;

  /**
   * Array. Contains the merge rule.
   */
  merges: string[];
}

/**
 * A GPT-2 tokenizer using Byte-Pair Encoding subword segmentation.
 *
 * This tokenizer class will tokenize raw strings into integer sequences and
 * is based on `BytePairTokenizer`. Unlike the underlying tokenizer, it will
 * check for all special tokens needed by GPT-2 models.
 *
 * This tokenizer does not provide truncation or padding of inputs.
 *
 * When given an input of a batch of strings (`tf.Tensor`), the layer will
 * output a `tf.Tensor[]`.
 *
 * Examples:
 *
 * ```js
 * const vocabulary = new Map([
 *    ['<|endoftext|>', 0], ['butter', 1], ['fly', 2]]);
 * const merges = ['b u', 't t', 'e r', 'bu tt', 'butt er', 'f l', 'fl y'];
 * const tokenizer = new BytePairTokenizer({vocabulary, merges});
 *
 * tokenizer.tokenize(tensor(['butterfly']))[0].print();
 * tokenizer.tokenize(tensor(['butterfly, butter<|endoftext|>']))[1].print();
 *
 * tokenizer.detokenize([tensor([1, 2, 0])]).print();
 */
export class GPT2Tokenizer extends BytePairTokenizer {
  private readonly _endTokenId: number;
  private readonly _startTokenId: number;
  private readonly _padTokenId: number;

  constructor(args: GPT2TokenizerArgs) {

    // Special tokens.
    const endToken = '<|endoftext|>';

    super({
      vocabulary: args.vocabulary,
      merges: args.merges,
      unsplittableTokens: [endToken]
    });

    // Check whether special tokens are present in the vocabulary.
    if (!this.vocabulary.includes(endToken)) {
      throw new ValueError(
        `Cannot find token '${endToken}' in the provided 'vocabulary'. Please` +
        ` provide '${endToken}' in your 'vocabulary' or use a pretrained` +
        ` 'vocabulary' name.`
      );
    }

    this._endTokenId = this.tokenToId(endToken);
    this._startTokenId = this._endTokenId;
    this._padTokenId = 0;
  }

  get endTokenId() {
    return this._endTokenId;
  }

  get startTokenId() {
    return this._startTokenId;
  }

  get padTokenId() {
    return this._padTokenId;
  }

  override getConfig(): serialization.ConfigDict {
    const config = super.getConfig();
    // In the constructor, we pass the list of special tokens to the
    // `unsplittableTokens` arg of the superclass' constructor. Hence, we
    // delete it from the config here.
    delete config.unsplittableTokens;
    return config;
  }
}
serialization.registerClass(GPT2Tokenizer);
