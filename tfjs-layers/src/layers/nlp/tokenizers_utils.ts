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
export class StaticHashTable<K, V extends number|string> {
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

  lookup(keys: Tensor[]): Tensor[] {
    const values = keys.map(t => {
      const innerValues: V[] = [];
      for (const key of t.dataSync() as unknown as K[]) {
        innerValues.push(this.get(key));
      }

      return tensor(innerValues, t.shape);
    });

    return values;
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
  // TODO(orderique): modify to use id2value map. Debug for correct behavior.
  private _cache: Map<string, string>;

  constructor() {
    this._cache = new Map();
  }

  /**
   * Insert token <=> encoded outputs pairs.
   */
  insert(
    keys: Tensor|string[], values: string[]): BytePairTokenizerCache {
    const arrKeys = keys instanceof Tensor ?
      keys.dataSync() as unknown as string[] : keys;

    arrKeys.forEach((key, idx) => this._cache.set(key, values[idx]));
    return this;
  }

  /**
   * Look up the encoded outputs of given tokens.
   */
  lookup(keys: Tensor|string[]): string[] {
    const arrKeys = keys instanceof Tensor ?
      keys.dataSync() as unknown as string[] : keys;
    return arrKeys.map(key => this._cache.get(key));
  }
}

/**
 * Remove certain strings from input tensor.
 */
export function removeStringsFromInputs(
  inputs: Tensor[], stringToRemove: string): Tensor[] {

  const stringArrInputs =
    inputs.map(input => input.dataSync() as unknown as string[]);

  const filteredStrArrays = stringArrInputs
    .map(arr => arr.filter(s => s !== stringToRemove))
    .filter(arr => arr.length > 0);

  const filteredTensors = filteredStrArrays.map(arr => tensor(arr));

  return filteredTensors;
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
const pL = 'a-zA-ZáàâäãåçéèêëíìîïñóòôöõúùûüýÿæœÁÀÂÄÃÅÇÉÈÊËÍÌÎÏÑÓÒÔÖÕÚÙÛÜÝŸÆŒĵ';
const pN = '0-9';
export const SPLIT_PATTERN_1 = new RegExp(
  `'s|'t|'re|'ve|'m|'ll|'d` +
  `|[\\s${SPECIAL_WHITESPACES.source}]+` +
  `[\\n\\r\\t\\f६${SPECIAL_WHITESPACES.source}]| ?${pL}+|`+
  ` ?${pN}+| ?[^\\s${pL}${pN}${SPECIAL_WHITESPACES.source}]+`,
  'gu'
);

const SPLIT_PATTERN_2 = new RegExp(`[\\s६${SPECIAL_WHITESPACES.source}]\$`);

function flatten<T>(inputs: T[][]): T[] {
  return inputs.reduce(
    (accumulator, value) => accumulator.concat(value), []);
}

export function regexSplit(
  strs: string[]|string[][],
  delimRegexPattern: RegExp | string,
  keepDelimRegexPattern?: RegExp | string): string[][] {

  if (strs[0] instanceof Array) {
    const mapped = strs.map(arr => regexSplit(
      arr as string[], delimRegexPattern, keepDelimRegexPattern));
    return mapped.map((doubleArr) => flatten(doubleArr));
  }

  strs = strs as string[];

  if (!(delimRegexPattern instanceof RegExp)) {
    if (keepDelimRegexPattern) {
      delimRegexPattern = new RegExp(
        `(${delimRegexPattern})`, 'g');
    }
    return strs.map(str => str.split(delimRegexPattern).filter(s => s));
  }

  let regexPattern = delimRegexPattern as unknown as RegExp;
  if (!regexPattern.flags.includes('g')) {
    regexPattern = new RegExp(regexPattern.source, regexPattern.flags + 'g');
  }

  return strs.map(str => {
    const matches = str.matchAll(regexPattern);

    const splitString = [];
    let currIdx = 0;
    for (const match of matches) {
      splitString.push(str.slice(currIdx, match.index));
      if (keepDelimRegexPattern) {
        splitString.push(
          str.slice(match.index, match.index! + match[0].length));
      }
      currIdx = match.index! + match[0].length;
    }
    if (currIdx !== str.length) {
      splitString.push(str.slice(currIdx, str.length));
    }
    return splitString.filter(s => s);
  });
}

export function tensorToArr<T>(input: Tensor): T[] {
  return input.dataSync() as unknown as T[];
}

export function tensorArrTo2DArr<T>(inputs: Tensor[]): T[][] {
  return inputs.map(input => tensorToArr<T>(input));
}

export function splitStringsForBpe(
  inputs: Tensor, unsplittableTokens?: string[]): Tensor[] {

  // We need to recreate the exact behavior of token presplitting in the
  // original gpt2 implementation which uses a lookahead. We are using an
  // alternative by inserting a special token "६" before leading space of
  // non-space characters and after the trailing space, e.g.,
  // " tf" will be "६ tf".
  const pattern1 = new RegExp(`( )([^\s${SPECIAL_WHITESPACES}])`);
  const pattern2 = new RegExp(`(\s${SPECIAL_WHITESPACES})\$`);

  const inputsStr = tensorToArr<string>(inputs).map(str =>
    str.replace(pattern1, `६$1$2`).replace(pattern2, `$1६`)
  );

  let alts: string[];
  let rawTokens: string[][];

  if (unsplittableTokens && unsplittableTokens.length > 0) {
    alts = createAltsForUnsplittableTokens(unsplittableTokens);
    unsplittableTokens.forEach((token, idx) => {
      const alt = alts[idx];
      const escapedToken = token.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');

      rawTokens = regexSplit(rawTokens !== undefined ?
        rawTokens : inputsStr, escapedToken, escapedToken);
      rawTokens = rawTokens.map(
        arr => arr.map(t => t.replace(escapedToken, alt)));
    });
  }
  rawTokens = regexSplit(rawTokens !== undefined ?
    rawTokens : inputsStr, SPLIT_PATTERN_1, SPLIT_PATTERN_1);
  // Second pass splits out the last whilespace char or "६".
  rawTokens  = regexSplit(rawTokens, SPLIT_PATTERN_2, SPLIT_PATTERN_2);

  if (unsplittableTokens && unsplittableTokens.length > 0) {
    // Replace special tokens alternate with originals.
    unsplittableTokens.forEach((token, idx) => {
      const alt = alts[idx];
      const escapedAlt = alt.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
      rawTokens = rawTokens.map(
        arr => arr.map(t => t.replace(escapedAlt, token)));
    });
  }

  return removeStringsFromInputs(rawTokens.map(tokens => tensor(tokens)), '६');
}
