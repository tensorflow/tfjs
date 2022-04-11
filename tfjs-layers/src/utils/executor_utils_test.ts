/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {LruCache} from '../utils/executor_utils';
import {describeMathCPU} from '../utils/test_utils';

// tslint:enable

describeMathCPU('LruCache', () => {
  it('Delete the leaset recent used entry when exceeding the size', () => {
    const maxEntries = 3;
    const cache = new LruCache<number>(maxEntries);
    cache.put('1', 1);  // Caching [1]
    cache.put('2', 2);  // Caching [1, 2]
    cache.put('3', 3);  // Caching [1, 2, 3]
    cache.get('1');     // Caching [2, 3, 1]
    cache.put('4', 4);  // Caching [3, 1, 4]

    expect(cache.get('2')).toBeUndefined();
    expect(cache.get('1')).toBe(1);
    expect(cache.get('3')).toBe(3);
    expect(cache.get('4')).toBe(4);
  });

  it('Reduce cache entries while decreasing maxEntries', () => {
    const cache = new LruCache<number>(100);
    for (let i = 0; i < 100; i++) {
      cache.put(i.toString(), i);
    }
    cache.setMaxEntries(99);

    // The first entry should be deleted, because the number of entries exceeded
    // the maxEntries.
    expect(cache.get('0')).toBeUndefined();
    expect(cache.get('1')).toBe(1);
    expect(cache.get('99')).toBe(99);
  });
});
