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
import { Tensor, serialization, tensor} from '@tensorflow/tfjs-core';
// import * as tfc from '@tensorflow/tfjs-core';

import { Layer, LayerArgs } from '../../engine/topology';
import { NotImplementedError, ValueError } from '../../errors';
import { BytePairTokenizerCache, StaticHashTable, bytesToUnicode, createStaticHashtable } from './tokenizers_utils';


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
 *    tokenize(inputs: Tensor): Tensor[] {
 *      const stringInputs = inputs.dataSync() as unknown as string[];
 *      return stringInputs.map(input => Tensor(input.split(' ')));
 *    }
 *
 *    override detokenize(inputs: Tensor[]): Tensor {
 *      const stringInputs = inputs.map(
 *        input => input.dataSync() as unknown as string[]);
 *      return Tensor(stringInputs.map(str => str.join(' ')));
 *    }
 *  }
 *
 * const tokenizer = new WhitespaceSplitterTokenizer();
 *
 * tokenizer.tokenize(Tensor(['this is a test']))[0].print();
 *
 * tokenizer.detokenize([Tensor(['this', 'is', 'a', 'test'])]).print();
 * ```
 */
export abstract class Tokenizer extends Layer {
  /**
   * Transform input tensors of strings into output tokens.
   *
   * @param inputs Input tensor.
   * @param kwargs Additional keyword arguments.
   */
  abstract tokenize(inputs: Tensor): Tensor[];

  /**
   * Transform tokens back into strings.
   *
   * @param inputs Input tensor.
   * @param kwargs Additional keyword arguments.
   */
  detokenize(inputs: Tensor[]): Tensor {
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
    inputs: Tensor|Tensor[],
    {mode = 'tokenize'}: TokenizerOptions={}
  ): Tensor|Tensor[] {

    if (mode === 'tokenize') {
      if (inputs instanceof Array) {
        throw new ValueError(`tokenize expects Tensor, not Tensor[].`);
      }
      return this.tokenize(inputs);
    }

    if (mode === 'detokenize') {
      if (!(inputs instanceof Array)) {
        throw new ValueError(`detokenize expects Tensor[], not Tensor.`);
      }
      return this.detokenize(inputs);
    }

    throw new ValueError(`Input mode=${mode} is not supported.`);
  }
}

/* Original source: keras-nlp/byte_pair_tokenizer.py */
// TODO(pforderique): Support filename string inputs for vocabulary and merges.
export declare interface BytePairTokenizerArgs extends LayerArgs {
  /**
   * Maps token to integer ids
   */
  vocabulary: Map<string, number>;

  /**
   * Array. Contains the merge rule.
   */
  merges: string[];

  /**
   * Integer. If set, the output will be padded or truncated to the
   * `sequenceLength`. Defaults to `null`.
   */
  sequenceLength?: number;

  /**
   * Boolean. Whether to add an initial space to the input. This tokenizer is
   * whitespace aware, and will tokenize a word with a leading space
   * differently. Adding a prefix space to the first word will cause it to be
   * tokenized equivalently to all subsequent words in the sequence.
   * Defaults to `false`.
   */
  addPrefixSpace?: boolean;

  /**
   * Array. A list of strings that will never be split during the word-level
   * splitting applied before the byte-pair encoding. This can be used to ensure
   * special tokens map to unique indices in the vocabulary, even if these
   * special tokens contain splittable characters such as punctuation. Special
   * tokens must still be included in `vocabulary`. Defaults to `None`.
   */
  unsplittableTokens?: string[];

  dtype?: 'string'|'int32';
}

export class BytePairTokenizer extends Tokenizer {
  /** @nocollapse */
  static readonly className = 'BytePairTokenizer';

  private _vocabulary: Map<string, number>;
  private merges: string[];

  private readonly sequenceLength: number;
  private readonly addPrefixSpace: boolean;
  private readonly unsplittableTokens: string[];
  private readonly _dtype: 'int32'|'string';

  private readonly byte2Unicode: StaticHashTable<number, string>;
  private readonly unicode2Byte: StaticHashTable<string, number>;
  private readonly cache = new BytePairTokenizerCache();

  private readonly tokenToIdMap: StaticHashTable<string, number>;
  private readonly idToTokenMap: StaticHashTable<number, string>;

  private readonly mergeRanksLookupDefault: number;
  private readonly mergeRanks: StaticHashTable<string, number>;

  constructor(args: BytePairTokenizerArgs) {
    super(args);

    this._vocabulary = new Map(args.vocabulary);
    this.merges = [...args.merges];

    this.sequenceLength = args.sequenceLength || null;
    this.addPrefixSpace = args.addPrefixSpace || false;
    this.unsplittableTokens = args.unsplittableTokens || null;
    this._dtype = args.dtype || 'int32';

    // Create byte <=> unicode mapping. This is useful for handling
    // whitespace tokens.
    const [byteList, unicodeList] = bytesToUnicode();
    this.byte2Unicode = createStaticHashtable(
      Array.from(byteList), unicodeList, '');
    this.unicode2Byte = createStaticHashtable(
      unicodeList, Array.from(byteList), -1);

    if (this.unsplittableTokens) {
      // Put unsplittable tokens into cache, so it won't be further split and
      // merged.
      this.cache.insert(this.unsplittableTokens, this.unsplittableTokens);
    }

    // Create mapping between string tokens to int ids, and vice versa.
    const bytePairs = [...this._vocabulary.keys()];
    const bytePairEncodingIndicies = [...this._vocabulary.values()];

    this.tokenToIdMap = createStaticHashtable(
      bytePairs, bytePairEncodingIndicies, -1);

    this.idToTokenMap = createStaticHashtable(
      bytePairEncodingIndicies, bytePairs, '');

    // Create ranking of merge rules, this is the same as order of merge pairs
    // in `this.merges`.
    this.mergeRanksLookupDefault = this.merges.length + 1;
    this.mergeRanks = createStaticHashtable(
      this.merges,
      [...Array(this.merges.length).keys()],
      this.mergeRanksLookupDefault
    );

    this._dtype;
    this.byte2Unicode; this.unicode2Byte; this.tokenToIdMap;
    this.idToTokenMap; this.mergeRanks;
  }

  /**
   * Get the tokenizer vocabulary as a list of string tokens.
   */
  override get vocabulary(): string[] {
    return [...this._vocabulary.keys()];
  }

  /**
   * Get the size of the tokenizer vocabulary.
   */
  override get vocabularySize(): number {
    return this._vocabulary.size;
  }

  /**
   * Convert an integer id to a string token.
   */
  override idToToken(id: number): string {
    // This will be slow, but keep memory usage down compared to building a
    // dict. Assuming the main use case is looking up a few special tokens
    // early in the vocab, this should be fine.
    const keys = this.vocabulary;
    for (const token of keys) {
      if (this._vocabulary.get(token) === id) {
        return token;
      }
    }
    return null;
  }

  /**
   * Convert a string token to an integer id.
   */
  override tokenToId(token: string): number {
    return this._vocabulary.get(token);
  }

  override getConfig(): serialization.ConfigDict {
    const config = {
      vocabulary: this.vocabulary,
      merges: this.merges,
      sequenceLength: this.sequenceLength,
      addPrefixSpace: this.addPrefixSpace,
      unsplittableTokens: this.unsplittableTokens,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  private async bpeMergeOneStep(
    words: Tensor[], mask: boolean[]): Promise<[Tensor[], boolean[]]> {

    // Notes (to be deleted):
    // words: string[][] like [[b'b', b'r', b'o', b'w', b'n'], [b'.'],
    // [b'b', b'l', b'a', b'c', b'k'], [b'.']]
    const wordsStr = await Promise.all(words.map(
      async t => t.data() as unknown as string[]));

    // Get all word pairs.
    const first = wordsStr.map(arr => arr.slice(0, -1));
    const second = wordsStr.map(arr => arr.slice(1, arr.length));

    // Mask empty.
    const nonEmptyMask = second.map(arr => arr.length > 0);
    mask = mask.map((a, idx) => a && nonEmptyMask[idx]);
    if (!mask.some(e => e)) {
      return [words, mask];
    }
    const nonEmptyIndices = mask
      .map((bool, idx) => bool ? idx : -1)
      .filter(e => e !== -1);

    const filteredFirst = nonEmptyIndices.map(idx => first[idx]);
    const filteredSecond = nonEmptyIndices.map(idx => second[idx]);

    // Get byte pair ranking in merge rules.
    const pairs: string[][] = filteredFirst.map((firstSubArr, idx) => {
      const secondSubArr = filteredSecond[idx];

      return firstSubArr.map(
        (char, idx) => [char, secondSubArr[idx]].join(' '));
    });
    const pairRanksTensor = await this.mergeRanks.lookup(
      pairs.map(arr => tensor(arr)));
    const pairRanks = await Promise.all(pairRanksTensor.map(
      async t => t.data() as unknown as number[]));

    // Get BPE pair ranks.
    const minPairRank = pairRanks.map(arr => Math.min(...arr));
    const pairFoundMask = minPairRank.map(
      rank => rank !== this.mergeRanksLookupDefault);

    // Tokens that cannot be further merged are marked as finished.
    nonEmptyIndices.forEach((index, idx) => {
      const update = pairFoundMask[idx];
      mask[index] = update;
    });
    if (!mask.some(e => e)) {
      return [words, mask];
    }

    return [[], []];
  }

  private async bpeMerge(words: Tensor[]): Promise<Tensor[]> {
    // Notes (to be deleted):
    // words: string[][] like [[b'b', b'r', b'o', b'w', b'n'], [b'.'],
    // [b'b', b'l', b'a', b'c', b'k'], [b'.']]
    this.bpeMergeAndUpdateCache;

    const numWords = words.length;

    // Merge bytes.
    function loopCondition(mask: boolean[]): boolean {
      return mask.some(e => e);
    }

    const initialMask: boolean[] = Array(numWords).fill(true);

    let mergedWords = words;
    let mask = initialMask;
    while (loopCondition(mask)) {
      [mergedWords, mask] = await this.bpeMergeOneStep(mergedWords, mask);
    }

    return mergedWords;
  }

  /**
   * Map token bytes to unicode using `byte2unicode`.
   */
  private async transformBytes(tokens: Tensor): Promise<Tensor[]> {
    const tokensStr = await tokens.data() as unknown as string[];

    const splitBytes = tokensStr.map(
      token => tensor(token.split('').map(c => c.charCodeAt(0))));
    const splitUnicode = await this.byte2Unicode.lookup(splitBytes);

    return splitUnicode;
  }

  /**
   * Process unseen tokens and add to cache.
   */
  private async bpeMergeAndUpdateCache(tokens: Tensor) {
    // tokens string[]
    const words = await this.transformBytes(tokens);
    const tokenizedWords = this.bpeMerge(words);
    tokenizedWords;
  }

  tokenize(inputs: Tensor): Tensor[] {
    const stringInputs = inputs.dataSync() as unknown as string[];
    return stringInputs.map(input => tensor(input.split(' ')));
  }

  override detokenize(inputs: Tensor[]): Tensor {
    const stringInputs = inputs.map(
      input => input.dataSync() as unknown as string[]);
    return tensor(stringInputs.map(str => str.join(' ')));
  }
}
serialization.registerClass(BytePairTokenizer);
