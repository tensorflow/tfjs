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

/* Original source: keras-nlp/byte_pair_tokenizer.py */

import { Tensor, tensor } from '@tensorflow/tfjs-core';
import { ValueError } from '../../errors';

export function bytesToUnicode(): [Uint8Array, string[]] {
  const inclusiveRange = (start: number, end: number) =>
    Array.from({ length: (end - start + 1) }, (v, k) => k + start);

  const bs = [
    ...inclusiveRange('!'.charCodeAt(0), '~'.charCodeAt(0)),
    ...inclusiveRange('¡'.charCodeAt(0), '¬'.charCodeAt(0)),
    ...inclusiveRange('®'.charCodeAt(0), 'ÿ'.charCodeAt(0))
  ];

  const cs = [...bs];
  let n = 0;

  // Removes mapping an int to a whitespace character
  for (let b = 0; b < 2 ** 8; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(2 ** 8 + n);
      n++;
    }
  }

  const chars = cs.map(n => String.fromCharCode(n));

  // TODO(orderique): Verify same functionality.
  const bytes = Uint8Array.from(bs);

  return [bytes, chars];
}

/**
 * StaticHashTable includes a `lookup` function for multiple keys at once.
 */
class StaticHashTable<K, V extends number|string> {
  private _map: Map<K, V>;

  constructor(keys: K[], values: V[], private readonly defaultValue: V) {
    if (keys.length !== values.length) {
      throw new ValueError(`keys and values arrays must be same length.
        Instead got lengths ${keys.length} and ${values.length}.`
      );
    }
    const keyValPairs: Array<[K, V]> = [];
    for (let idx = 0; idx < keys.length; idx++) {
      const key = keys[idx];
      const val = values[idx];
      keyValPairs.push([key, val]);
    }

    this._map = new Map(keyValPairs);
  }

  get(key: K): V {
    if (this._map.has(key)) {
      return this._map.get(key);
    }
    return this.defaultValue;
  }

  async lookup(keys: Tensor[]): Promise<Tensor[]> {
    const values = keys.map(async t => {
      const innerValues: V[] = [];
      for (const key of await t.data() as unknown as K[]) {
        innerValues.push(this.get(key));
      }

      return tensor(innerValues, t.shape);
    });

    return Promise.all(values);
  }
}

export function createStaticHashtable<K, V extends number|string>(
  keys: K[], values: V[], defaultVal: V): StaticHashTable<K, V> {

  return new StaticHashTable(keys, values, defaultVal);
}

/**
 * Cache that stores the encoded result of seen tokens.
 *
 * The cache key is string tensor or python strings, and the value is split
 * tokens joined by whitespace. For example, "dragonfly" => "dragon fly"
 *
 * Examples:
 *
 * ```js
 * const cache = new BytePairTokenizerCache();
 * cache.insert(["butterfly", "dragonfly"], ["but ter fly", "dragon fly"]);
 * cache.lookup(["butterfly"]);
 * ```
 */
export class BytePairTokenizerCache {
  private _cache: Map<string, string>;

  constructor() {
    this._cache = new Map();
  }

  /**
   * Insert token <=> encoded outputs pairs.
   */
  async insert(
    keys: Tensor|string[], values: string[]): Promise<BytePairTokenizerCache> {
    const arrKeys = keys instanceof Tensor ?
      await keys.data() as unknown as string[] : keys;

    arrKeys.forEach((key, idx) => this._cache.set(key, values[idx]));
    return this;
  }

  /**
   * Look up the encoded outputs of given tokens.
   */
  async lookup(keys: Tensor|string[]): Promise<string[]> {
    const arrKeys = keys instanceof Tensor ?
      await keys.data() as unknown as string[] : keys;
    return arrKeys.map(key => this._cache.get(key));
  }
}

/**
 * Remove certain strgins from input tensor.
 */
export async function removeStringsFromInputs(
  inputs: Tensor[], stringToRemove: string): Promise<Tensor[]> {
    const filteredInputs = inputs.map(async input => tensor(
        (await input.data() as unknown as string[])
          .filter(str => str !== stringToRemove),
        input.shape));

  return Promise.all(filteredInputs);
}

/**
 * Create alternates for all special tokens that will be not split during
 * tokenization.
 */
export function createAltsForUnsplittableTokens(
  unsplittableTokens: string[]): string[] {

  const prefix = 'ĵ';

  // Trim out splitters.
  const replacePattern: RegExp = /'|\s+|[^\p{L}\p{N}]+/gu;
  return unsplittableTokens.map(
    token => prefix + token.replace(replacePattern, ''));
}

// Typescript and TF handles special spaces differently, we need to
// manually handle special spaces during string split.
const SPECIAL_WHITESPACES = /\u00A0\u2009\u202f\u3000/;

// String splitting regex pattern.
const SPLIT_PATTERN_1 = new RegExp(
  `'s|'t|'re|'ve|'m|'ll|'d`
  + `|[\\s<specialSpaces>]+[\\n\\r\\t\\f६<specialSpaces>]| ?\\p{L}+|`
  + ' ?[\\p{N}]+| ?[^\\s\p{L}\\p{N}<specialSpaces>]+'.replace(
    '<specialSpaces>', SPECIAL_WHITESPACES.source),
  'gu'
);

const SPLIT_PATTERN_2 = new RegExp(`[\s६${SPECIAL_WHITESPACES.source}]\$`);

export function regexSplit(
  strs: string[],
  delimRegexPattern: RegExp|string,
  keepDelimRegexPattern?: RegExp|string): string[][] {

  if (keepDelimRegexPattern) {
    delimRegexPattern = delimRegexPattern instanceof RegExp ?
      delimRegexPattern.source : delimRegexPattern;
    delimRegexPattern = new RegExp(`(${delimRegexPattern})`, 'g');
  }

  return strs.map(str => str.split(delimRegexPattern).filter(s => s));
}

export async function splitStringForBpe(
  inputs: Tensor, unsplittableTokens?: string[]): Promise<Tensor[]> {

  // We need to recreate the exact behavior of token presplitting in the
  // original gpt2 implementation which uses a lookahead. We are using an
  // alternative by inserting a special token "६" before leading space of
  // non-space characters and after the trailing space, e.g.,
  // " tf" will be "६ tf".
  const pattern1 = new RegExp(`( )([^\s${SPECIAL_WHITESPACES}])`);
  const pattern2 = new RegExp(`(\s${SPECIAL_WHITESPACES})\$`);

  let inputsStr = (await inputs.data() as unknown as string[]).map(str =>
    str.replace(pattern1, `६$1$2`).replace(pattern2, `$1६`)
  );

  if (unsplittableTokens && unsplittableTokens.length > 0) {
    const alts = createAltsForUnsplittableTokens(unsplittableTokens);
    unsplittableTokens.forEach((token, idx) => {
      const escapedToken = token.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
      SPLIT_PATTERN_1; SPLIT_PATTERN_2; inputsStr; alts; idx; escapedToken;
    });
  }

  return [];
}
