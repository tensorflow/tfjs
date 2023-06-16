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

import { Tensor, tensor1d } from '@tensorflow/tfjs-core';
import { ValueError } from '../../errors';

export function bytesToUnicode(): [Uint8Array, string[]] {
  const range = (start: number, end: number) =>
    Array.from({ length: (end - start + 1) }, (v, k) => k + start);

  const bs = [
    ...range('!'.charCodeAt(0), '~'.charCodeAt(0)),
    ...range('¡'.charCodeAt(0), '¬'.charCodeAt(0)),
    ...range('®'.charCodeAt(0), 'ÿ'.charCodeAt(0))
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
 * StaticHashTable extends Map and includes a `lookup` function that
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
    return this._map.get(key) || this.defaultValue;
  }

  lookup(keys: Tensor[]): Tensor[] {
    const values = keys.map(tensor => {
      const innerValues: V[] = [];
      for (const key of tensor.dataSync() as unknown as K[]) {
        innerValues.push(this.get(key));
      }

      return tensor1d(innerValues as string[]|number[]);
    });

    return values;
  }
}

export function createStaticHashtable<T1, T2 extends number|string>(
  keys: T1[], values: T2[], defaultVal: T2): StaticHashTable<T1, T2> {

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
 * ```
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
  insert(keys: Tensor|string[], values: string[]): BytePairTokenizerCache {
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
 * Remove certain strgins from input tensor.
 */
export function removeStringsFromInputs(
  inputs: Tensor[], stringToRemove: string): Tensor[] {
    const filteredInputs = inputs
      .map(input => tensor1d(
        (input.dataSync() as unknown as string[])
        .filter(str => str != stringToRemove)));

  return filteredInputs;
}

/**
 * Create alternates for all special tokens that will be not split during
 * tokenization.
 */
export function createAltsForUnsplittableTokens(
  unsplittableTokens: string[]): string[] {

  const prefix = 'ĵ';

  // Trim out splitters.
  const replacePattern: RegExp = /'|\s+|[^\p{L}\p{N}]+/;
  return unsplittableTokens.map(
    token => prefix + token.replace(replacePattern, ''));
}
