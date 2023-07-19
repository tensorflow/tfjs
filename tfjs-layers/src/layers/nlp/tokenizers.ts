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
import { Tensor, serialization, tensor, tidy} from '@tensorflow/tfjs-core';

import { Layer, LayerArgs } from '../../engine/topology';
import { NotImplementedError, ValueError } from '../../errors';
import { BytePairTokenizerCache, StaticHashTable, bytesToUnicode, createStaticHashtable, removeStringsFromInputs, splitStringsForBpe } from './tokenizers_utils';
import { tensorToArr, tensorArrTo2DArr } from './utils';

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
 * tokenizer.tokenize(tensor(['this is a test']))[0].print();
 *
 * tokenizer.detokenize([tensor(['this', 'is', 'a', 'test'])]).print();
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
}

/**
 * Byte-pair encoding tokenizer layer.
 *
 * This BPE tokenizer provides the same functionality as the official GPT-2
 * tokenizer. Given the same `vocabulary` which maps tokens to ids, and `merges`
 * which describes BPE merge rules, it should provide the same output as OpenAI
 * implementation (https://github.com/openai/gpt-2/blob/master/src/encoder.py).
 *
 * If input is a batch of strings (rank > 0):
 * By default, the layer will output a `Tensor[]`.
 * If `sequenceLength` is set, the layer will output a `Tensor[]` where all
 * inputs have been padded or truncated to `sequenceLength`.
 *
 * Examples:
 *
 * Tokenize
 * ```js
 * const vocabulary = new Map([['butter', 1], ['fly', 2]]);
 * const merges = ['b u', 't t', 'e r', 'bu tt', 'butt er', 'f l', 'fl y'];
 * const tokenizer = new BytePairTokenizer({vocabulary, merges});
 *
 * tokenizer.tokenize(tensor(['butterfly']))[0].print();
 * tokenizer.tokenize(tensor(['butterfly, butter']))[1].print();
 * ```
 *
 * Detokenize
 * ```js
 * const vocabulary = new Map([['butter', 1], ['fly', 2]]);
 * const merges = ['b u', 't t', 'e r', 'bu tt', 'butt er', 'f l', 'fl y'];
 * const tokenizer = new BytePairTokenizer({vocabulary, merges});
 *
 * tokenizer.detokenize([[1, 2]]).print();
 * ```
 */
export class BytePairTokenizer extends Tokenizer {
  /** @nocollapse */
  static readonly className = 'BytePairTokenizer';

  private _vocabulary: Map<string, number>;
  private merges: string[];

  private readonly sequenceLength: number;
  private readonly addPrefixSpace: boolean;
  private readonly unsplittableTokens: string[];

  private readonly byte2Unicode: StaticHashTable<number, string>;
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

    // Create byte <=> unicode mapping. This is useful for handling
    // whitespace tokens.
    const [byteList, unicodeList] = bytesToUnicode();
    this.byte2Unicode = createStaticHashtable(
      Array.from(byteList), unicodeList, '');

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
  override idToToken(id: number): string | undefined {
    // This will be slow, but keep memory usage down compared to building a
    // dict. Assuming the main use case is looking up a few special tokens
    // early in the vocab, this should be fine.
    const keys = this.vocabulary;
    for (const token of keys) {
      if (this._vocabulary.get(token) === id) {
        return token;
      }
    }
    return undefined;
  }

  /**
   * Convert a string token to an integer id.
   */
  override tokenToId(token: string): number | undefined {
    return this._vocabulary.get(token);
  }

  override getConfig(): serialization.ConfigDict {
    const config = {
      vocabulary: Array.from(this._vocabulary.entries()),
      merges: this.merges,
      sequenceLength: this.sequenceLength,
      addPrefixSpace: this.addPrefixSpace,
      unsplittableTokens: this.unsplittableTokens,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  /**
   * Perform one step of byte-pair merge.
   */
  private bpeMergeOneStep(
    words: Tensor[], mask: boolean[]): [Tensor[], boolean[]] {

    const wordsStr = tensorArrTo2DArr(words) as string[][];

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

      return firstSubArr.map((char, idx) => `${char} ${secondSubArr[idx]}`);
    });
    const pairRanksTensor = this.mergeRanks.lookup(
      pairs.map(arr => tensor(arr)));
    const pairRanks = tensorArrTo2DArr(pairRanksTensor) as number[][];

    // Get BPE pair ranks.
    const minPairRank = pairRanks.map(
      arr => arr.reduce((a, b) => Math.min(a, b), Infinity));
    const pairFoundMask = minPairRank.map(
      rank => rank !== this.mergeRanksLookupDefault);

    // Tokens that cannot be further merged are marked as finished.
    for (const [idx, index] of nonEmptyIndices.entries()) {
      const update = pairFoundMask[idx];
      mask[index] = update;
    }
    if (!mask.some(e => e)) {
      return [words, mask];
    }

    function argMin(arr: number[]): number {
      return arr.indexOf(arr.reduce((a, b) => Math.min(a, b), Infinity));
    }

    const maskedPairRanks = pairRanks.filter((_, idx) => pairFoundMask[idx]);
    const minPairRankIndices = maskedPairRanks.map(arr => argMin(arr));

    // Get words and pairs to process.
    const unfinishedWords = wordsStr.filter((_, idx) => mask[idx]);

    const pairLeft = unfinishedWords.map(
      (word, idx) => word[minPairRankIndices[idx]]);

    const pairRight = unfinishedWords.map(
      (word, idx) => word[minPairRankIndices[idx] + 1]);

    const mergedPairs = pairLeft.map((left, idx) => {
      const right = pairRight[idx];
      return `${left}${right}`;
    });
    const unfinishedWordsIndices = mask
      .map((_, idx) => idx)
      .filter((_, idx) => mask[idx]);

    const mergedPairIndices = unfinishedWordsIndices.map(
      (index, idx) => [index, minPairRankIndices[idx]]);
    const emptyStringIndices = unfinishedWordsIndices.map(
      (index, idx) => [index, minPairRankIndices[idx] + 1]);

    for (const [idx, indices] of mergedPairIndices.entries()) {
      const [wordIdx, charIdx] = indices;
      const mergedPair = mergedPairs[idx];
      wordsStr[wordIdx][charIdx] = mergedPair;
    }

    for (const indices of emptyStringIndices) {
      const [wordIdx, charIdx] = indices;
      wordsStr[wordIdx][charIdx] = '';
    }

    words = wordsStr.map(word => tensor(word));
    words = removeStringsFromInputs(words, '');

    return [words, mask];
  }

  /**
   * Perform byte-pair merge for each word in the inputs.
   */
  private bpeMerge(words: Tensor[]): Tensor[] {
    const numWords = words.length;

    // Merge bytes.
    function loopCondition(mask: boolean[]): boolean {
      return mask.some(e => e);
    }

    const initialMask: boolean[] = Array(numWords).fill(true);

    let mergedWords = words;
    let mask = initialMask;
    while (loopCondition(mask)) {
      [mergedWords, mask] = this.bpeMergeOneStep(mergedWords, mask);
    }

    return mergedWords;
  }

  /**
   * Map token bytes to unicode using `byte2unicode`.
   */
  private transformBytes(tokens: Tensor): Tensor[] {
    const tokensStr = tensorToArr(tokens) as string[];

    const splitBytes = tokensStr.map(
      token => tensor(token.split('').map(c => c.charCodeAt(0))));
    const splitUnicode = this.byte2Unicode.lookup(splitBytes);

    return splitUnicode;
  }

  /**
   * Process unseen tokens and add to cache.
   */
  private bpeMergeAndUpdateCache(tokens: Tensor) {
    const words = this.transformBytes(tokens);
    const tokenizedWordsTensor = this.bpeMerge(words);
    const tokenizedWords = tensorArrTo2DArr(tokenizedWordsTensor) as string[][];

    // For each word, join all its token by a whitespace,
    // e.g., ["dragon", "fly"] => "dragon fly" for hash purpose.
    const joinedTokens = tokenizedWords.map(word => word.join(' '));

    this.cache.insert(tokens, joinedTokens);
  }

  tokenize(inputs: Tensor): Tensor[] {
    return tidy(() => {
      if (this.addPrefixSpace) {
        const strInputs = tensorToArr(inputs) as string[];
        inputs = tensor(strInputs.map(word => ' ' + word));
      }

      const rawTokensTensor =
        splitStringsForBpe(inputs, this.unsplittableTokens);
      const rawTokens = tensorArrTo2DArr(rawTokensTensor) as string[][];

      const tokenRowSplits = [0];
      for (const [idx, token] of rawTokens.entries()) {
        tokenRowSplits.push(tokenRowSplits[idx] + token.length);
      }

      const flatTokens = rawTokens.reduce((acc, e) => acc.concat(e), []);

      // Check cache.
      const cacheLookup = this.cache.lookup(flatTokens);
      const cacheMask = cacheLookup.map(e => e === '');

      const hasUnseenWords = cacheMask.some(
        (bool, idx) => bool && flatTokens[idx] !== '');

      const processUnseenTokens = (): string[]  => {
        const unseenTokens = flatTokens.filter((_, idx) => cacheMask[idx]);
        this.bpeMergeAndUpdateCache(tensor(unseenTokens));
        return this.cache.lookup(flatTokens);
      };

      // If `has_unseen_words == True`, it means not all tokens are in cache,
      // we will process the unseen tokens. Otherwise return the cache lookup.
      const tokenizedWords =
        hasUnseenWords ? processUnseenTokens() : cacheLookup;

      const tokensTensor = this.tokenToIdMap.lookup(
        tokenizedWords.map(word => tensor(word.split(' '))));
      const tokens = tokensTensor.map(t => Array.from(t.dataSync()));

      // Unflatten to match input.
      const newTokenRowSplits = [0];
      for (const [idx, token] of tokens.entries()) {
        newTokenRowSplits.push(newTokenRowSplits[idx] + token.length);
      }
      const newFlatTokens = tokens.reduce((acc, e) => acc.concat(e), []);
      const gatheredIndices =
        tokenRowSplits.map(index => newTokenRowSplits[index]);

      let tokens2D: Tensor[] = [];
      for (let i = 0; i < gatheredIndices.length - 1; i++) {
        const [start, end] = [gatheredIndices[i], gatheredIndices[i+1]];
        const row = newFlatTokens.slice(start, end);
        tokens2D.push(tensor(row));
      }

      // Convert to a dense output if `sequenceLength` is set.
      if (this.sequenceLength) {
        // pad or truncate
        tokens2D = tokens2D.map(t => {
          if (t.size === this.sequenceLength) {
            return t;
          } else if (t.size > this.sequenceLength) {
            return t.slice(0, this.sequenceLength);
          } else {
            return t.pad([[0, this.sequenceLength - t.size]]);
          }
        });
      }

      return tokens2D;
    });
  }

  override detokenize(inputs: Tensor[]): Tensor {
    const unicodeText = this.idToTokenMap.lookup(inputs)
      .map(t => (tensorToArr(t) as string[]).join(''));

    return tensor(unicodeText);
  }
}
serialization.registerClass(BytePairTokenizer);
