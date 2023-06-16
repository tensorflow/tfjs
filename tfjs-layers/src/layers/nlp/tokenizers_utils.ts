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
 * HashTable extends Map and includes a `lookup` function that
 */
class HashTable<K, V extends number|string> extends Map<K, V> {
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

    super(keyValPairs);
  }

  override get(key: K) {
    if (!this.has(key)) {
      this.set(key, this.defaultValue);
    }
    return super.get(key);
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

export function createHashtable<T1, T2 extends number|string>(
  keys: T1[], values: T2[], defaultVal: T2): HashTable<T1, T2> {

  return new HashTable(keys, values, defaultVal);
}
