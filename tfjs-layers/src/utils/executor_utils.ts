/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/**
 * LruCache: A mapping from the String to T. If the number of the entries is
 * exceeding the `maxEntries`, the LruCache will delete the least recently
 * used entry.
 */

import {env} from '@tensorflow/tfjs-core';

export class LruCache<T> {
  private cache: Map<string, T>;
  private maxEntries: number;

  constructor(maxEntries?: number) {
    this.maxEntries =
        maxEntries || env().getNumber('TOPOLOGICAL_SORT_CACHE_MAX_ENTRIES');
    this.cache = new Map<string, T>();
  }

  /**
   * Get the entry for the key and mark it as used recently.
   */
  public get(key: string): T {
    let entry: T;
    if (this.cache.has(key)) {
      entry = this.cache.get(key);
      this.cache.delete(key);
      this.cache.set(key, entry);
    }
    return entry;
  }

  /**
   * Put the entry into the cache. If the key already existed, mark the key as
   * used recently.
   */
  public put(key: string, value: T) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxEntries) {
      const keyToDelete = this.cache.keys().next().value;
      this.cache.delete(keyToDelete);
    }
    this.cache.set(key, value);
  }
}
