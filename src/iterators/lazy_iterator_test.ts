/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 *
 * =============================================================================
 */

import {DataElement, DataElementArray, DataElementObject} from '../types';
import {iteratorFromIncrementing, iteratorFromZipped, LazyIterator, ZipMismatchMode} from './lazy_iterator';
import {iteratorFromConcatenated} from './lazy_iterator';
import {iteratorFromConcatenatedFunction} from './lazy_iterator';
import {iteratorFromFunction, iteratorFromItems} from './lazy_iterator';

export class TestIntegerIterator extends LazyIterator<number> {
  currentIndex = 0;
  data: number[];

  constructor(protected readonly length = 100) {
    super();
    this.data = Array.from({length}, (v, k) => k);
  }

  summary() {
    return `TestIntegers`;
  }

  async next(): Promise<IteratorResult<number>> {
    if (this.currentIndex >= this.length) {
      return {value: null, done: true};
    }
    const result = this.data[this.currentIndex];
    this.currentIndex++;
    // Sleep for a millisecond every so often.
    // This purposely scrambles the order in which these promises are resolved,
    // to demonstrate that the various methods still process the stream
    // in the correct order.
    if (Math.random() < 0.1) {
      await new Promise(res => setTimeout(res, 1));
    }
    return {value: result, done: false};
  }
}

describe('LazyIterator', () => {
  it('collects all stream elements into an array', async () => {
    const readIterator = new TestIntegerIterator();
    const result = await readIterator.collect();
    expect(result.length).toEqual(100);
  });

  it('reads chunks in order', async () => {
    const readIterator = new TestIntegerIterator();
    const result = await readIterator.collect();
    expect(result.length).toEqual(100);
    for (let i = 0; i < 100; i++) {
      expect(result[i]).toEqual(i);
    }
  });

  it('filters elements', async () => {
    const readIterator = new TestIntegerIterator().filter(x => x % 2 === 0);
    const result = await readIterator.collect();
    expect(result.length).toEqual(50);
    for (let i = 0; i < 50; i++) {
      expect(result[i]).toEqual(2 * i);
    }
  });

  it('maps elements', async () => {
    const readIterator = new TestIntegerIterator().map(x => `item ${x}`);
    const result = await readIterator.collect();
    expect(result.length).toEqual(100);
    for (let i = 0; i < 100; i++) {
      expect(result[i]).toEqual(`item ${i}`);
    }
  });

  it('maps elements through an async function', async () => {
    const readIterator = new TestIntegerIterator().mapAsync(async x => {
      // Sleep for a millisecond every so often.
      // This purposely scrambles the order in which these promises are
      // resolved, to demonstrate that we still process the
      // stream in the correct order.
      if (Math.random() < 0.1) {
        await new Promise(res => setTimeout(res, 1));
      }
      return Promise.resolve(`item ${x}`);
    });
    // It's important to prefetch in order to test the promise randomization
    // above.  Note collect() already prefetches by default, but here we do it
    // explicitly anyway just to be extra clear.
    const result = await readIterator.prefetch(200).collect();
    expect(result.length).toEqual(100);
    for (let i = 0; i < 100; i++) {
      expect(result[i]).toEqual(`item ${i}`);
    }
  });

  it('flatmaps simple elements', async () => {
    const readStream = new TestIntegerIterator().flatmap(
        x => [`item ${x} A`, `item ${x} B`, `item ${x} C`]);
    const result = await readStream.collect();
    expect(result.length).toEqual(300);
    for (let i = 0; i < 100; i++) {
      expect(result[3 * i + 0]).toEqual(`item ${i} A`);
      expect(result[3 * i + 1]).toEqual(`item ${i} B`);
      expect(result[3 * i + 2]).toEqual(`item ${i} C`);
    }
  });

  it('flatmap flattens object elements but not their contents', async () => {
    const readStream = new TestIntegerIterator().flatmap(
        x =>
            [{foo: `foo ${x} A`, bar: `bar ${x} A`},
             {foo: `foo ${x} B`, bar: `bar ${x} B`},
             {foo: `foo ${x} C`, bar: `bar ${x} C`},
    ]);
    const result = await readStream.collect()
    expect(result.length).toEqual(300);
    for (let i = 0; i < 100; i++) {
      expect(result[3 * i + 0]).toEqual({foo: `foo ${i} A`, bar: `bar ${i} A`});
      expect(result[3 * i + 1]).toEqual({foo: `foo ${i} B`, bar: `bar ${i} B`});
      expect(result[3 * i + 2]).toEqual({foo: `foo ${i} C`, bar: `bar ${i} C`});
    }
  });

  it('flatmap flattens array elements but not their contents', async () => {
    const readStream = new TestIntegerIterator().flatmap(
        x => [
            [`foo ${x} A`, `bar ${x} A`],
            [`foo ${x} B`, `bar ${x} B`],
            [`foo ${x} C`, `bar ${x} C`],
    ]);
    const result = await readStream.collect();
    expect(result.length).toEqual(300);
    for (let i = 0; i < 100; i++) {
      expect(result[3 * i + 0]).toEqual([`foo ${i} A`, `bar ${i} A`]);
      expect(result[3 * i + 1]).toEqual([`foo ${i} B`, `bar ${i} B`]);
      expect(result[3 * i + 2]).toEqual([`foo ${i} C`, `bar ${i} C`]);
    }
  });

  it('batches elements to a row-major representation', async () => {
    const readIterator = new TestIntegerIterator().rowMajorBatch(8);
    const result = await readIterator.collect();
    expect(result.length).toEqual(13);
    for (let i = 0; i < 12; i++) {
      expect(result[i]).toEqual(Array.from({length: 8}, (v, k) => (i * 8) + k));
    }
    expect(result[12]).toEqual([96, 97, 98, 99]);
  });

  it('batches elements to a column-major representation', async () => {
    const readIterator = new TestIntegerIterator().columnMajorBatch(8);
    const result = await readIterator.collect();
    expect(result.length).toEqual(13);
    for (let i = 0; i < 12; i++) {
      expect(result[i]).toEqual(Array.from({length: 8}, (v, k) => (i * 8) + k));
    }
    expect(result[12]).toEqual([96, 97, 98, 99]);
  });

  it('can be limited to a certain number of elements', async () => {
    const readIterator = new TestIntegerIterator().take(8);
    const result = await readIterator.collect();
    expect(result).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });

  it('is unaltered by a negative or undefined take() count.', async () => {
    const baseIterator = new TestIntegerIterator();
    const readIterator = baseIterator.take(-1);
    const result = await readIterator.collect();
    expect(result).toEqual(baseIterator.data);
    const baseIterator2 = new TestIntegerIterator();
    const readIterator2 = baseIterator2.take(undefined);
    const result2 = await readIterator2.collect();
    expect(result2).toEqual(baseIterator2.data);
  });

  it('can skip a certain number of elements', async () => {
    const readIterator = new TestIntegerIterator().skip(88).take(8);
    const result = await readIterator.collect();
    expect(result).toEqual([88, 89, 90, 91, 92, 93, 94, 95]);
  });

  it('is unaltered by a negative or undefined skip() count.', async () => {
    const baseIterator = new TestIntegerIterator();
    const readIterator = baseIterator.skip(-1);
    const result = await readIterator.collect();
    expect(result).toEqual(baseIterator.data);
    const baseIterator2 = new TestIntegerIterator();
    const readIterator2 = baseIterator2.skip(undefined);
    const result2 = await readIterator2.collect();
    expect(result2).toEqual(baseIterator2.data);
  });

  it('can selectively ignore upstream errors', async () => {
    const readIterator = new TestIntegerIterator().take(10).map(x => {
      if (x % 2 === 0) {
        throw new Error('Oh no, an even number!');
      }
      return x;
    });
    // The 'true' response means the iterator should continue.
    const errorIgnoringIterator = readIterator.handleErrors((e) => true);
    const result = await errorIgnoringIterator.collect();
    expect(result).toEqual([1, 3, 5, 7, 9]);
  });

  it('can terminate cleanly based on upstream errors', async () => {
    const readIterator = new TestIntegerIterator().map(x => {
      if (x % 2 === 0) {
        throw new Error(`Oh no, an even number: ${x}`);
      }
      return x;
    });
    // The 'true' response means the iterator should continue.
    // But in the case of 10, return false, terminating the stream.
    const errorHandlingIterator = readIterator.handleErrors(
        (e) => e.message !== 'Oh no, an even number: 10');
    const result = await errorHandlingIterator.collect();
    expect(result).toEqual([1, 3, 5, 7, 9]);
  });

  it('can selectively propagate upstream errors', async () => {
    const readIterator = new TestIntegerIterator().map(x => {
      if (x % 2 === 0) {
        throw new Error(`Oh no, an even number: ${x}`);
      }
      return x;
    });
    const errorHandlingIterator = readIterator.handleErrors((e) => {
      if (e.message === 'Oh no, an even number: 2') {
        throw e;
      }
      return true;
    });
    try {
      await errorHandlingIterator.collect();
    } catch (e) {
      expect(e.message).toEqual('Oh no, an even number: 2');
    }
  });

  it('can be forced to execute serially', async () => {
    // This is like FilterIterator, except that it does not enforce serial
    // execution.  For testing.
    class ParallelFilterIterator<T> extends LazyIterator<T> {
      constructor(
          protected upstream: LazyIterator<T>,
          protected predicate: (value: T) => boolean) {
        super();
      }

      summary() {
        return `${this.upstream.summary} -> ParallelFilterIterator`;
      }

      async next(): Promise<IteratorResult<T>> {
        while (true) {
          const item = await this.upstream.next();
          if (item.done || this.predicate(item.value)) {
            return item;
          }
        }
      }
    }

    const newReadIterator = () =>
        new ParallelFilterIterator((new TestIntegerIterator()).take(10), x => {
          return x % 2 === 0;
        });

    // Without enforcing serial execution, order gets scrambled; consequently
    // the done signal arrives before any of the accepted elements!
    const badResult = await newReadIterator().collect();
    expect(badResult).toEqual([0]);

    // But with serial execution, everything is fine.
    const goodResult = await newReadIterator().serial().collect();
    expect(goodResult).toEqual([0, 2, 4, 6, 8]);
  });

  it('can be created from an array', async () => {
    const readIterator = iteratorFromItems([1, 2, 3, 4, 5, 6]);
    const result = await readIterator.collect();
    expect(result).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('can be created from a function', async () => {
    let i = -1;
    const func = () =>
        ++i < 7 ? {value: i, done: false} : {value: null, done: true};

    const readIterator = iteratorFromFunction(func);
    const result = await readIterator.collect();
    expect(result).toEqual([0, 1, 2, 3, 4, 5, 6]);
  });

  it('can be created with incrementing integers', async () => {
    const readIterator = iteratorFromIncrementing(0).take(7);
    const result = await readIterator.collect();
    expect(result).toEqual([0, 1, 2, 3, 4, 5, 6]);
  });

  it('can be concatenated', async () => {
    const a = iteratorFromItems([1, 2, 3]);
    const b = iteratorFromItems([4, 5, 6]);
    const readIterator = a.concatenate(b);
    const result = await readIterator.collect();
    expect(result).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('can be created by concatenating streams', async () => {
    const a = new TestIntegerIterator();
    const b = new TestIntegerIterator();
    const readIterator = iteratorFromConcatenated(iteratorFromItems([a, b]));
    const result = await readIterator.collect();
    expect(result.length).toEqual(200);
  });

  it('can be created by concatenating streams from a function', async () => {
    const readIterator = iteratorFromConcatenatedFunction(
        () => ({value: new TestIntegerIterator(), done: false}), 3);
    const expectedResult: number[] = [];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 100; j++) {
        expectedResult[i * 100 + j] = j;
      }
    }

    const result = await readIterator.collect();
    expect(result).toEqual(expectedResult);
  });

  it('can be created by zipping an array of streams', async done => {
    try {
      const a = new TestIntegerIterator();
      const b = new TestIntegerIterator().map(x => x * 10);
      const c = new TestIntegerIterator().map(x => `string ${x}`);
      const readStream = iteratorFromZipped([a, b, c]);
      const result = await readStream.collect();
      expect(result.length).toEqual(100);

      // each result has the form [x, x * 10, 'string ' + x]

      for (const e of result) {
        const ee = e as DataElementArray;
        expect(ee[1]).toEqual(ee[0] as number * 10);
        expect(ee[2]).toEqual(`string ${ee[0]}`);
      }
      done();
    } catch (e) {
      done.fail();
    }
  });

  it('can be created by zipping a dict of streams', async done => {
    try {
      const a = new TestIntegerIterator();
      const b = new TestIntegerIterator().map(x => x * 10);
      const c = new TestIntegerIterator().map(x => `string ${x}`);
      const readStream = iteratorFromZipped({a, b, c});
      const result = await readStream.collect();
      expect(result.length).toEqual(100);

      // each result has the form {a: x, b: x * 10, c: 'string ' + x}

      for (const e of result) {
        const ee = e as DataElementObject;
        expect(ee['b']).toEqual(ee['a'] as number * 10);
        expect(ee['c']).toEqual(`string ${ee['a']}`);
      }
      done();
    } catch (e) {
      done.fail();
    }
  });

  it('can be created by zipping a nested structure of streams', async done => {
    try {
      const a = new TestIntegerIterator().map(x => ({'a': x, 'constant': 12}));
      const b = new TestIntegerIterator().map(
          x => ({'b': x * 10, 'array': [x * 100, x * 200]}));
      const c = new TestIntegerIterator().map(x => ({'c': `string ${x}`}));
      const readStream = iteratorFromZipped([a, b, c]);
      const result = await readStream.collect();
      expect(result.length).toEqual(100);

      // each result has the form
      // [
      //   {a: x, 'constant': 12}
      //   {b: x * 10, 'array': [x * 100, x * 200]},
      //   {c: 'string ' + x}
      // ]

      for (const e of result) {
        const ee = e as DataElementArray;
        const aa = ee[0] as DataElementObject;
        const bb = ee[1] as DataElementObject;
        const cc = ee[2] as DataElementObject;
        expect(aa['constant']).toEqual(12);
        expect(bb['b']).toEqual(aa['a'] as number * 10);
        expect(bb['array']).toEqual([
          aa['a'] as number * 100, aa['a'] as number * 200
        ]);
        expect(cc['c']).toEqual(`string ${aa['a']}`);
      }
      done();
    } catch (e) {
      done.fail();
    }
  });

  it('zip requires streams of the same length by default', async done => {
    try {
      const a = new TestIntegerIterator(10);
      const b = new TestIntegerIterator(3);
      const c = new TestIntegerIterator(2);
      const readStream = iteratorFromZipped([a, b, c]);
      await readStream.collect();
      // expected error due to default ZipMismatchMode.FAIL
      done.fail();
    } catch (e) {
      done();
    }
  });

  it('zip can be told to terminate when the shortest stream terminates',
     async done => {
       try {
         const a = new TestIntegerIterator(10);
         const b = new TestIntegerIterator(3);
         const c = new TestIntegerIterator(2);
         const readStream =
             iteratorFromZipped([a, b, c], ZipMismatchMode.SHORTEST);
         const result = await readStream.collect();
         expect(result.length).toEqual(2);
         done();
       } catch (e) {
         done.fail();
       }
     });

  it('zip can be told to terminate when the longest stream terminates',
     async done => {
       try {
         const a = new TestIntegerIterator(10);
         const b = new TestIntegerIterator(3);
         const c = new TestIntegerIterator(2);
         const readStream =
             iteratorFromZipped([a, b, c], ZipMismatchMode.LONGEST);
         const result = await readStream.collect();
         expect(result.length).toEqual(10);
         expect(result[9]).toEqual([9, null, null]);
         done();
       } catch (e) {
         done.fail();
       }
     });

  /**
   * This test demonstrates behavior that is intrinsic to the tf.data zip()
   * API, but that may not be what users ultimately want when zipping dicts.
   * This may merit a convenience function (e.g., maybe flatZip()).
   */
  it('zipping DataElement streams requires manual merge', async done => {
    function naiveMerge(xs: DataElement[]): DataElement {
      const result = {};
      for (const x of xs) {
        // For now, we do nothing to detect name collisions here
        Object.assign(result, x);
      }
      return result;
    }

    try {
      const a = new TestIntegerIterator().map(x => ({'a': x}));
      const b = new TestIntegerIterator().map(x => ({'b': x * 10}));
      const c = new TestIntegerIterator().map(x => ({'c': `string ${x}`}));
      const zippedStream = iteratorFromZipped([a, b, c]);
      // At first, each result has the form
      // [{a: x}, {b: x * 10}, {c: 'string ' + x}]

      const readStream =
          zippedStream.map(e => naiveMerge(e as DataElementArray));
      // Now each result has the form {a: x, b: x * 10, c: 'string ' + x}

      const result = await readStream.collect();
      expect(result.length).toEqual(100);

      for (const e of result) {
        const ee = e as DataElementObject;
        expect(ee['b']).toEqual(ee['a'] as number * 10);
        expect(ee['c']).toEqual(`string ${ee['a']}`);
      }
      done();
    } catch (e) {
      done.fail();
    }
  });
});
