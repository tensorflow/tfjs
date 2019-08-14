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

import * as tf from '@tensorflow/tfjs-core';
import * as seedrandom from 'seedrandom';

import {IteratorContainer} from '../types';
import {deepClone} from '../util/deep_clone';
import {deepMapAndAwaitAll, DeepMapAsyncResult, DeepMapResult, deepZip, zipToList} from '../util/deep_map';
import {GrowingRingBuffer} from '../util/growing_ring_buffer';
import {RingBuffer} from '../util/ring_buffer';

// Here we implement a simple asynchronous iterator.
// This lets us avoid using either third-party stream libraries or
// recent TypeScript language support requiring polyfills.

/**
 * Create a `LazyIterator` from an array of items.
 */
export function iteratorFromItems<T>(items: T[]): LazyIterator<T> {
  return new ArrayIterator(items);
}

/**
 * Create a `LazyIterator` of incrementing integers.
 */
export function iteratorFromIncrementing(start: number): LazyIterator<number> {
  let i = start;
  return iteratorFromFunction(() => ({value: i++, done: false}));
}

/**
 * Create a `LazyIterator` from a function.
 *
 * ```js
 * let i = -1;
 * const func = () =>
 *    ++i < 5 ? {value: i, done: false} : {value: null, done: true};
 * const iter = tf.data.iteratorFromFunction(func);
 * await iter.forEachAsync(e => console.log(e));
 * ```
 *
 * @param func A function that produces data on each call.
 */
export function iteratorFromFunction<T>(
    func: () =>
        IteratorResult<T>| Promise<IteratorResult<T>>): LazyIterator<T> {
  return new FunctionCallIterator(func);
}

/**
 * Create a `LazyIterator` by concatenating underlying streams, which are
 * themselves provided as a stream.
 *
 * This can also be thought of as a "stream flatten" operation.
 *
 * @param baseIterators A stream of streams to be concatenated.
 * @param baseErrorHandler An optional function that can intercept `Error`s
 *   raised during a `next()` call on the base stream.  This function can decide
 *   whether the error should be propagated, whether the error should be
 *   ignored, or whether the base stream should be terminated.
 */
export function iteratorFromConcatenated<T>(
    baseIterators: LazyIterator<LazyIterator<T>>,
    baseErrorHandler?: (e: Error) => boolean): LazyIterator<T> {
  return new ChainedIterator(baseIterators, baseErrorHandler);
}

/**
 * Create a `LazyIterator` by concatenating streams produced by calling a
 * stream-generating function a given number of times.
 *
 * Since a `LazyIterator` is read-once, it cannot be repeated, but this
 * function can be used to achieve a similar effect:
 *
 *   LazyIterator.ofConcatenatedFunction(() => new MyIterator(), 6);
 *
 * @param iteratorFunc: A function that produces a new stream on each call.
 * @param count: The number of times to call the function.
 * @param baseErrorHandler An optional function that can intercept `Error`s
 *   raised during a `next()` call on the base stream.  This function can decide
 *   whether the error should be propagated, whether the error should be
 *   ignored, or whether the base stream should be terminated.
 */
export function iteratorFromConcatenatedFunction<T>(
    iteratorFunc: () => IteratorResult<LazyIterator<T>>, count: number,
    baseErrorHandler?: (e: Error) => boolean): LazyIterator<T> {
  return iteratorFromConcatenated(
      iteratorFromFunction(iteratorFunc).take(count), baseErrorHandler);
}

/**
 * Create a `LazyIterator` by zipping together an array, dict, or nested
 * structure of `LazyIterator`s (and perhaps additional constants).
 *
 * The underlying streams must provide elements in a consistent order such
 * that they correspond.
 *
 * Typically, the underlying streams should have the same number of
 * elements. If they do not, the behavior is determined by the
 * `mismatchMode` argument.
 *
 * The nested structure of the `iterators` argument determines the
 * structure of elements in the resulting iterator.
 *
 * @param iterators: An array or object containing LazyIterators at the
 * leaves.
 * @param mismatchMode: Determines what to do when one underlying iterator
 * is exhausted before the others.  `ZipMismatchMode.FAIL` (the default)
 * causes an error to be thrown in this case.  `ZipMismatchMode.SHORTEST`
 * causes the zipped iterator to terminate with the furst underlying
 * streams, so elements remaining on the longer streams are ignored.
 * `ZipMismatchMode.LONGEST` causes the zipped stream to continue, filling
 * in nulls for the exhausted streams, until all streams are exhausted.
 */
export function iteratorFromZipped<O extends tf.TensorContainer>(
    iterators: IteratorContainer,
    mismatchMode: ZipMismatchMode = ZipMismatchMode.FAIL): LazyIterator<O> {
  return new ZipIterator<O>(iterators, mismatchMode);
}

/**
 * An asynchronous iterator, providing lazy access to a potentially
 * unbounded stream of elements.
 *
 * Iterator can be obtained from a dataset:
 * `const iter = await dataset.iterator();`
 */
export abstract class LazyIterator<T> {
  // This class implements AsyncIterator<T>, but we have not yet set the
  // TypeScript --downlevelIteration flag to enable that.

  abstract summary(): string;

  /**
   * Returns a `Promise` for the next element in the stream.
   *
   * When an item can be provided successfully, the return value is
   * `{value:T, done:false}`.
   *
   * Calling next() on a closed stream returns `{value:null, done:true}`.
   */
  abstract async next(): Promise<IteratorResult<T>>;

  /**
   * Collect all remaining elements of a bounded stream into an array.
   * Obviously this will succeed only for small streams that fit in memory.
   * Useful for testing.
   *
   * @returns A Promise for an array of stream elements, which will resolve
   *   when the stream is exhausted.
   */
  async toArray(): Promise<T[]> {
    const result: T[] = [];
    let x = await this.next();
    while (!x.done) {
      result.push(x.value);
      x = await this.next();
    }
    return result;
  }

  /**
   * Collect all elements of this dataset into an array with prefetching 100
   * elements. This is useful for testing, because the prefetch changes the
   * order in which the Promises are resolved along the processing pipeline.
   * This may help expose bugs where results are dependent on the order of
   * Promise resolution rather than on the logical order of the stream (i.e.,
   * due to hidden mutable state).
   *
   * @returns A Promise for an array of stream elements, which will resolve
   *   when the stream is exhausted.
   */
  async toArrayForTest(): Promise<T[]> {
    const stream = this.prefetch(100);
    const result: T[] = [];
    let x = await stream.next();
    while (!x.done) {
      result.push(x.value);
      x = await stream.next();
    }
    return result;
  }

  /**
   * Draw items from the stream until it is exhausted.
   *
   * This can be useful when the stream has side effects but no output.  In
   * that case, calling this function guarantees that the stream will be
   * fully processed.
   */
  async resolveFully(): Promise<void> {
    let x = await this.next();
    while (!x.done) {
      x = await this.next();
    }
  }

  /**
   * Draw items from the stream until it is exhausted, or a predicate fails.
   *
   * This can be useful when the stream has side effects but no output.  In
   * that case, calling this function guarantees that the stream will be
   * fully processed.
   */
  async resolveWhile(predicate: (r: T) => boolean): Promise<void> {
    let x = await this.next();
    let shouldContinue = predicate(x.value);
    while ((!x.done) && shouldContinue) {
      x = await this.next();
      shouldContinue = predicate(x.value);
    }
  }

  /**
   * Handles errors thrown on this stream using a provided handler function.
   *
   * @param handler A function that handles any `Error` thrown during a `next()`
   *   call and returns true if the stream should continue (dropping the failed
   *   call) or false if the stream should quietly terminate.  If the handler
   *   itself throws (or rethrows) an `Error`, that will be propagated.
   *
   * @returns A `LazyIterator` of elements passed through from upstream,
   *   possibly filtering or terminating on upstream `next()` calls that
   *   throw an `Error`.
   */
  handleErrors(handler: (error: Error) => boolean): LazyIterator<T> {
    return new ErrorHandlingLazyIterator(this, handler);
  }

  // TODO(soergel): Implement reduce() etc.

  /**
   * Filters this stream according to `predicate`.
   *
   * @param predicate A function mapping a stream element to a boolean or a
   * `Promise` for one.
   *
   * @returns A `LazyIterator` of elements for which the predicate was true.
   */
  filter(predicate: (value: T) => boolean): LazyIterator<T> {
    return new FilterIterator(this, predicate);
  }

  /**
   * Maps this stream through a 1-to-1 transform.
   *
   * @param transform A function mapping a stream element to a transformed
   *   element.
   *
   * @returns A `LazyIterator` of transformed elements.
   */
  map<O>(transform: (value: T) => O): LazyIterator<O> {
    return new MapIterator(this, transform);
  }

  /**
   * Maps this stream through an async 1-to-1 transform.
   *
   * @param transform A function mapping a stream element to a `Promise` for a
   *   transformed stream element.
   *
   * @returns A `LazyIterator` of transformed elements.
   */
  mapAsync<O>(transform: (value: T) => Promise<O>): LazyIterator<O> {
    return new AsyncMapIterator(this, transform);
  }

  /**
   * Maps this stream through a 1-to-1 transform, forcing serial execution.
   *
   * @param transform A function mapping a stream element to a transformed
   *   element.
   *
   * @returns A `LazyIterator` of transformed elements.
   */
  serialMapAsync<O>(transform: (value: T) => Promise<O>): LazyIterator<O> {
    return new AsyncMapIterator(this, transform).serial();
  }

  /**
   * Maps this stream through a 1-to-many transform.
   *
   * @param transform A function mapping a stream element to an array of
   *   transformed elements.
   *
   * @returns A `DataStream` of transformed elements.
   */
  flatmap<O>(transform: (value: T) => O[]): LazyIterator<O> {
    return new FlatmapIterator(this, transform);
  }

  /**
   * Apply a function to every element of the stream.
   *
   * @param f A function to apply to each stream element.
   */
  async forEachAsync(f: (value: T) => void): Promise<void> {
    return this.map(f).resolveFully();
  }

  /**
   * Apply a function to every element of the stream, forcing serial execution.
   *
   * @param f A function to apply to each stream element.  Should return 'true'
   *   to indicate that the stream should continue, or 'false' to cause it to
   *   terminate.
   */
  async serialForEach(f: (value: T) => Promise<boolean>): Promise<void> {
    return this.serialMapAsync(f).resolveWhile(x => (x === true));
  }

  /**
   * Groups elements into batches, represented as arrays of elements.
   *
   * We can think of the elements of this iterator as 'rows' (even if they are
   * nested structures).  By the same token, consecutive values for a given
   * key within the elements form a 'column'.  This matches the usual sense of
   * 'row' and 'column' when processing tabular data (e.g., parsing a CSV).
   *
   * Thus, "Row-major" means that the resulting batch is simply a collection of
   * rows: `[row1, row2, row3, ...]`.  This is contrast to the column-major
   * form, which is needed for vectorized computation.
   *
   * @param batchSize The number of elements desired per batch.
   * @param smallLastBatch Whether to emit the final batch when it has fewer
   *   than batchSize elements. Default true.
   * @returns A `LazyIterator` of batches of elements, represented as arrays
   *   of the original element type.
   */
  rowMajorBatch(batchSize: number, smallLastBatch = true): LazyIterator<T[]> {
    return new RowMajorBatchIterator(this, batchSize, smallLastBatch);
  }

  /**
   * Groups elements into batches, represented in column-major form.
   *
   * We can think of the elements of this iterator as 'rows' (even if they are
   * nested structures).  By the same token, consecutive values for a given
   * key within the elements form a 'column'.  This matches the usual sense of
   * 'row' and 'column' when processing tabular data (e.g., parsing a CSV).
   *
   * Thus, "column-major" means that the resulting batch is a (potentially
   * nested) structure representing the columns.  Each column entry, then,
   * contains a collection of the values found in that column for a range of
   * input elements.  This representation allows for vectorized computation, in
   * contrast to the row-major form.
   *
   * The inputs should all have the same nested structure (i.e., of arrays and
   * dicts).  The result is a single object with the same nested structure,
   * where the leaves are arrays collecting the values of the inputs at that
   * location (or, optionally, the result of a custom function applied to those
   * arrays).
   *
   * @param batchSize The number of elements desired per batch.
   * @param smallLastBatch Whether to emit the final batch when it has fewer
   *   than batchSize elements. Default true.
   * @param zipFn: (optional) A function that expects an array of elements at a
   *   single node of the object tree, and returns a `DeepMapResult`.  The
   *   `DeepMapResult` either provides a result value for that node (i.e.,
   *   representing the subtree), or indicates that the node should be processed
   *   recursively.  The default zipFn recurses as far as possible and places
   *   arrays at the leaves.
   * @returns A `LazyIterator` of batches of elements, represented as an object
   *   with collections at the leaves.
   */
  columnMajorBatch(
      batchSize: number, smallLastBatch = true,
      // tslint:disable-next-line:no-any
      zipFn: (xs: any[]) => DeepMapResult = zipToList):
      LazyIterator<tf.TensorContainer> {
    // First collect the desired number of input elements as a row-major batch.
    const rowBatches = this.rowMajorBatch(batchSize, smallLastBatch);
    // Now 'rotate' or 'pivot' the data, collecting all values from each column
    // in the batch (i.e., for each key within the elements) into an array.
    return rowBatches.map(x => deepZip(x, zipFn));
  }

  /**
   * Concatenate this `LazyIterator` with another.
   *
   * @param iterator A `LazyIterator` to be concatenated onto this one.
   * @param baseErrorHandler An optional function that can intercept `Error`s
   *   raised during a `next()` call on the base stream.  This function can
   *   decide whether the error should be propagated, whether the error should
   *   be ignored, or whether the base stream should be terminated.
   * @returns A `LazyIterator`.
   */
  concatenate(
      iterator: LazyIterator<T>,
      baseErrorHandler?: (e: Error) => boolean): LazyIterator<T> {
    return new ChainedIterator(
        iteratorFromItems([this, iterator]), baseErrorHandler);
  }

  /**
   * Limits this stream to return at most `count` items.
   *
   * @param count The maximum number of items to provide from the stream. If
   * a negative or undefined value is given, the entire stream is returned
   *   unaltered.
   */
  take(count: number): LazyIterator<T> {
    if (count < 0 || count == null) {
      return this;
    }
    return new TakeIterator(this, count);
  }

  /**
   * Skips the first `count` items in this stream.
   *
   * @param count The number of items to skip.  If a negative or undefined
   * value is given, the entire stream is returned unaltered.
   */
  skip(count: number): LazyIterator<T> {
    if (count < 0 || count == null) {
      return this;
    }
    return new SkipIterator(this, count);
  }

  /**
   * Prefetch the first `bufferSize` items in this stream.
   *
   * Note this prefetches Promises, but makes no guarantees about when those
   * Promises resolve.
   *
   * @param bufferSize: An integer specifying the number of elements to be
   *   prefetched.
   */
  prefetch(bufferSize: number): LazyIterator<T> {
    return new PrefetchIterator(this, bufferSize);
  }

  // TODO(soergel): deep sharded shuffle, where supported

  /**
   * Randomly shuffles the elements of this stream.
   *
   * @param bufferSize: An integer specifying the number of elements from
   * this stream from which the new stream will sample.
   * @param seed: (Optional.) An integer specifying the random seed that
   * will be used to create the distribution.
   */
  shuffle(windowSize: number, seed?: string): LazyIterator<T> {
    return new ShuffleIterator(this, windowSize, seed);
  }

  /**
   * Force an iterator to execute serially: each next() call will await the
   * prior one, so that they cannot execute concurrently.
   */
  serial(): LazyIterator<T> {
    return new SerialIterator(this);
  }
}

// ============================================================================
// The following private classes serve to implement the chainable methods
// on LazyIterator.  Unfortunately they can't be placed in separate files,
// due to resulting trouble with circular imports.
// ============================================================================

// Iterators that just extend LazyIterator directly
// ============================================================================

class ArrayIterator<T> extends LazyIterator<T> {
  private trav = 0;
  constructor(protected items: T[]) {
    super();
  }

  summary() {
    return `Array of ${this.items.length} items`;
  }

  async next(): Promise<IteratorResult<T>> {
    if (this.trav >= this.items.length) {
      return {value: null, done: true};
    }
    const item = this.items[this.trav];
    this.trav++;
    return {value: deepClone(item), done: false};
  }
}

class FunctionCallIterator<T> extends LazyIterator<T> {
  constructor(
      protected nextFn: () => IteratorResult<T>| Promise<IteratorResult<T>>) {
    super();
  }

  summary() {
    return `Function call`;
  }

  async next(): Promise<IteratorResult<T>> {
    try {
      return this.nextFn();
    } catch (e) {
      // Modify the error message but leave the stack trace intact
      e.message =
          `Error thrown while iterating through a dataset: ${e.message}`;
      throw e;
    }
  }
}

class SerialIterator<T> extends LazyIterator<T> {
  // Strict Promise execution order:
  // a next() call may not even begin until the previous one completes.
  private lastRead: Promise<IteratorResult<T>>;

  constructor(protected upstream: LazyIterator<T>) {
    super();
    this.lastRead = Promise.resolve({value: null, done: false});
  }

  summary() {
    return `${this.upstream.summary()} -> Serial`;
  }

  async next(): Promise<IteratorResult<T>> {
    // This sets this.lastRead to a new Promise right away, as opposed to
    // saying `await this.lastRead; this.lastRead = this.serialNext();` which
    // would not work because this.nextRead would be updated only after the
    // promise resolves.
    this.lastRead = this.lastRead.then(() => this.serialNext());
    return this.lastRead;
  }

  private async serialNext(): Promise<IteratorResult<T>> {
    return this.upstream.next();
  }
}

class SkipIterator<T> extends LazyIterator<T> {
  // Strict Promise execution order:
  // a next() call may not even begin until the previous one completes.
  private lastRead: Promise<IteratorResult<T>>;

  // Local state that should not be clobbered by out-of-order execution.
  count = 0;

  constructor(protected upstream: LazyIterator<T>, protected maxCount: number) {
    super();
    this.lastRead = Promise.resolve({value: null, done: false});
  }

  summary() {
    return `${this.upstream.summary()} -> Skip`;
  }

  async next(): Promise<IteratorResult<T>> {
    // This sets this.lastRead to a new Promise right away, as opposed to
    // saying `await this.lastRead; this.lastRead = this.serialNext();` which
    // would not work because this.nextRead would be updated only after the
    // promise resolves.
    this.lastRead = this.lastRead.then(() => this.serialNext());
    return this.lastRead;
  }

  private async serialNext(): Promise<IteratorResult<T>> {
    // TODO(soergel): consider tradeoffs of reading in parallel, eg.
    // collecting next() promises in an Array and then waiting for
    // Promise.all() of those. Benefit: pseudo-parallel execution.  Drawback:
    // maybe delayed GC.
    while (this.count++ < this.maxCount) {
      const skipped = await this.upstream.next();
      // short-circuit if upstream is already empty
      if (skipped.done) {
        return skipped;
      }
      tf.dispose(skipped.value as {});
    }
    return this.upstream.next();
  }
}

class TakeIterator<T> extends LazyIterator<T> {
  count = 0;
  constructor(protected upstream: LazyIterator<T>, protected maxCount: number) {
    super();
  }

  summary() {
    return `${this.upstream.summary()} -> Take`;
  }

  async next(): Promise<IteratorResult<T>> {
    if (this.count++ >= this.maxCount) {
      return {value: null, done: true};
    }
    return this.upstream.next();
  }
}

// Note this batch just groups items into row-wise element arrays.
// Rotating these to a column-wise representation happens only at the dataset
// level.
class RowMajorBatchIterator<T> extends LazyIterator<T[]> {
  // Strict Promise execution order:
  // a next() call may not even begin until the previous one completes.
  private lastRead: Promise<IteratorResult<T[]>>;

  constructor(
      protected upstream: LazyIterator<T>, protected batchSize: number,
      protected enableSmallLastBatch = true) {
    super();
    this.lastRead = Promise.resolve({value: null, done: false});
  }

  summary() {
    return `${this.upstream.summary()} -> RowMajorBatch`;
  }

  async next(): Promise<IteratorResult<T[]>> {
    // This sets this.lastRead to a new Promise right away, as opposed to
    // saying `await this.lastRead; this.lastRead = this.serialNext();` which
    // would not work because this.nextRead would be updated only after the
    // promise resolves.
    this.lastRead = this.lastRead.then(() => this.serialNext());
    return this.lastRead;
  }

  private async serialNext(): Promise<IteratorResult<T[]>> {
    const batch: T[] = [];
    while (batch.length < this.batchSize) {
      const item = await this.upstream.next();
      if (item.done) {
        if (this.enableSmallLastBatch && batch.length > 0) {
          return {value: batch, done: false};
        }
        return {value: null, done: true};
      }
      batch.push(item.value);
    }
    return {value: batch, done: false};
  }
}

class FilterIterator<T> extends LazyIterator<T> {
  // Strict Promise execution order:
  // a next() call may not even begin until the previous one completes.
  private lastRead: Promise<IteratorResult<T>>;

  constructor(
      protected upstream: LazyIterator<T>,
      protected predicate: (value: T) => boolean) {
    super();
    this.lastRead = Promise.resolve({value: null, done: false});
  }

  summary() {
    return `${this.upstream.summary()} -> Filter`;
  }

  async next(): Promise<IteratorResult<T>> {
    // This sets this.lastRead to a new Promise right away, as opposed to
    // saying `await this.lastRead; this.lastRead = this.serialNext();` which
    // would not work because this.nextRead would be updated only after the
    // promise resolves.
    this.lastRead = this.lastRead.then(() => this.serialNext());
    return this.lastRead;
  }

  private async serialNext(): Promise<IteratorResult<T>> {
    while (true) {
      const item = await this.upstream.next();
      if (item.done || this.predicate(item.value)) {
        return item;
      }
      tf.dispose(item.value as {});
    }
  }
}

class MapIterator<I, O> extends LazyIterator<O> {
  constructor(
      protected upstream: LazyIterator<I>,
      protected transform: (value: I) => O) {
    super();
  }

  summary() {
    return `${this.upstream.summary()} -> Map`;
  }

  async next(): Promise<IteratorResult<O>> {
    const item = await this.upstream.next();
    if (item.done) {
      return {value: null, done: true};
    }
    const inputTensors = tf.tensor_util.getTensorsInContainer(item.value as {});
    // Careful: the transform may mutate the item in place.
    // That's why we have to remember the input Tensors above, and then
    // below dispose only those that were not passed through to the output.
    // Note too that the transform function is responsible for tidying
    // any intermediate Tensors.  Here we are concerned only about the
    // inputs.
    const mapped = this.transform(item.value);
    const outputTensors = tf.tensor_util.getTensorsInContainer(mapped as {});

    // TODO(soergel) faster intersection
    // TODO(soergel) move to tf.disposeExcept(in, out)?
    for (const t of inputTensors) {
      if (!tf.tensor_util.isTensorInList(t, outputTensors)) {
        t.dispose();
      }
    }
    return {value: mapped, done: false};
  }
}

class ErrorHandlingLazyIterator<T> extends LazyIterator<T> {
  count = 0;
  constructor(
      protected upstream: LazyIterator<T>,
      protected handler: (error: Error) => boolean) {
    super();
    this.lastRead = Promise.resolve({value: null, done: false});
  }

  summary() {
    return `${this.upstream.summary()} -> handleErrors`;
  }

  // Strict Promise execution order:
  // a next() call may not even begin until the previous one completes.
  private lastRead: Promise<IteratorResult<T>>;

  async next(): Promise<IteratorResult<T>> {
    // This sets this.lastRead to a new Promise right away, as opposed to
    // saying `await this.lastRead; this.lastRead = this.serialNext();` which
    // would not work because this.nextRead would be updated only after the
    // promise resolves.
    this.lastRead = this.lastRead.then(() => this.serialNext());
    return this.lastRead;
  }

  async serialNext(): Promise<IteratorResult<T>> {
    while (true) {
      try {
        return await this.upstream.next();
      } catch (e) {
        if (!this.handler(e)) {
          return {value: null, done: true};
        }
        // If the handler returns true, loop and fetch the next upstream item.

        // If the upstream iterator throws an endless stream of errors, and if
        // the handler says to ignore them, then we loop forever here.  That is
        // the correct behavior-- it's up to the handler to decide when to stop.
      }
    }
  }
}

class AsyncMapIterator<I, O> extends LazyIterator<O> {
  constructor(
      protected upstream: LazyIterator<I>,
      protected transform: (value: I) => Promise<O>) {
    super();
  }

  summary() {
    return `${this.upstream.summary()} -> AsyncMap`;
  }

  async next(): Promise<IteratorResult<O>> {
    const item = await this.upstream.next();
    if (item.done) {
      return {value: null, done: true};
    }
    const inputTensors = tf.tensor_util.getTensorsInContainer(item.value as {});
    // Careful: the transform may mutate the item in place.
    // That's why we have to remember the input Tensors above, and then
    // below dispose only those that were not passed through to the output.
    // Note too that the transform function is responsible for tidying
    // any intermediate Tensors.  Here we are concerned only about the
    // inputs.
    const mapped = await this.transform(item.value);
    const outputTensors = tf.tensor_util.getTensorsInContainer(mapped as {});

    // TODO(soergel) faster intersection
    // TODO(soergel) move to tf.disposeExcept(in, out)?
    for (const t of inputTensors) {
      if (!tf.tensor_util.isTensorInList(t, outputTensors)) {
        t.dispose();
      }
    }
    return {value: mapped, done: false};
  }
}

// Iterators that maintain a queue of pending items
// ============================================================================

/**
 * A base class for transforming streams that operate by maintaining an
 * output queue of elements that are ready to return via next().  This is
 * commonly required when the transformation is 1-to-many:  A call to next()
 * may trigger a call to the underlying stream, which will produce many
 * mapped elements of this stream-- of which we need to return only one, so
 * we have to queue the rest.
 */
export abstract class OneToManyIterator<T> extends LazyIterator<T> {
  // Strict Promise execution order:
  // a next() call may not even begin until the previous one completes.
  private lastRead: Promise<IteratorResult<T>>;

  // Local state that should not be clobbered by out-of-order execution.
  protected outputQueue: RingBuffer<T>;

  constructor() {
    super();
    this.outputQueue = new GrowingRingBuffer<T>();
    this.lastRead = Promise.resolve({value: null, done: false});
  }

  async next(): Promise<IteratorResult<T>> {
    // This sets this.lastRead to a new Promise right away, as opposed to
    // saying `await this.lastRead; this.lastRead = this.serialNext();` which
    // would not work because this.nextRead would be updated only after the
    // promise resolves.
    this.lastRead = this.lastRead.then(() => this.serialNext());
    return this.lastRead;
  }

  /**
   * Read one or more chunks from upstream and process them, possibly
   * reading or writing a carryover, and adding processed items to the
   * output queue.  Note it's possible that no items are added to the queue
   * on a given pump() call, even if the upstream stream is not closed
   * (e.g., because items are filtered).
   *
   * @return `true` if any action was taken, i.e. fetching items from the
   *   upstream source OR adding items to the output queue.  `false` if the
   *   upstream source is exhausted AND nothing was added to the queue
   * (i.e., any remaining carryover).
   */
  protected abstract async pump(): Promise<boolean>;

  async serialNext(): Promise<IteratorResult<T>> {
    // Fetch so that the queue contains at least one item if possible.
    // If the upstream source is exhausted, AND there are no items left in
    // the output queue, then this stream is also exhausted.
    while (this.outputQueue.length() === 0) {
      // TODO(soergel): consider parallel reads.
      if (!await this.pump()) {
        return {value: null, done: true};
      }
    }
    return {value: this.outputQueue.shift(), done: false};
  }
}
class FlatmapIterator<I, O> extends OneToManyIterator<O> {
  constructor(
      protected upstream: LazyIterator<I>,
      protected transform: (value: I) => O[]) {
    super();
  }

  summary() {
    return `${this.upstream.summary()} -> Flatmap`;
  }

  async pump(): Promise<boolean> {
    const item = await this.upstream.next();
    if (item.done) {
      return false;
    }
    const inputTensors = tf.tensor_util.getTensorsInContainer(item.value as {});
    // Careful: the transform may mutate the item in place.
    // that's why we have to remember the input Tensors above, and then
    // below dispose only those that were not passed through to the output.
    // Note too that the transform function is responsible for tidying any
    // intermediate Tensors.  Here we are concerned only about the inputs.
    const mappedArray = this.transform(item.value);
    const outputTensors =
        tf.tensor_util.getTensorsInContainer(mappedArray as {});
    this.outputQueue.pushAll(mappedArray);

    // TODO(soergel) faster intersection, and deduplicate outputTensors
    // TODO(soergel) move to tf.disposeExcept(in, out)?
    for (const t of inputTensors) {
      if (!tf.tensor_util.isTensorInList(t, outputTensors)) {
        t.dispose();
      }
    }

    return true;
  }
}

/**
 * Provides a `LazyIterator` that concatenates a stream of underlying
 * streams.
 *
 * Doing this in a concurrency-safe way requires some trickery.  In
 * particular, we want this stream to return the elements from the
 * underlying streams in the correct order according to when next() was
 * called, even if the resulting Promises resolve in a different order.
 */
export class ChainedIterator<T> extends LazyIterator<T> {
  // Strict Promise execution order:
  // a next() call may not even begin until the previous one completes.
  private lastRead: Promise<IteratorResult<T>> = null;

  // Local state that should not be clobbered by out-of-order execution.
  private iterator: LazyIterator<T> = null;
  private moreIterators: LazyIterator<LazyIterator<T>>;

  constructor(
      iterators: LazyIterator<LazyIterator<T>>,
      private readonly baseErrorHandler?: (e: Error) => boolean) {
    super();
    this.moreIterators = iterators;
  }

  summary() {
    const upstreamSummaries = 'TODO: fill in upstream of chained summaries';
    return `${upstreamSummaries} -> Chained`;
  }

  async next(): Promise<IteratorResult<T>> {
    this.lastRead = this.readFromChain(this.lastRead);
    return this.lastRead;
  }

  private async readFromChain(lastRead: Promise<IteratorResult<T>>):
      Promise<IteratorResult<T>> {
    // Must await on the previous read since the previous read may have advanced
    // the stream of streams, from which we need to read.
    // This is unfortunate since we can't parallelize reads. Which means
    // prefetching of chained streams is a no-op.
    // One solution is to prefetch immediately upstream of this.
    await lastRead;
    if (this.iterator == null) {
      const iteratorResult = await this.moreIterators.next();
      if (iteratorResult.done) {
        // No more streams to stream from.
        return {value: null, done: true};
      }
      this.iterator = iteratorResult.value;
      if (this.baseErrorHandler != null) {
        this.iterator = this.iterator.handleErrors(this.baseErrorHandler);
      }
    }
    const itemResult = await this.iterator.next();
    if (itemResult.done) {
      this.iterator = null;
      return this.readFromChain(lastRead);
    }
    return itemResult;
  }
}

export enum ZipMismatchMode {
  FAIL,      // require zipped streams to have the same length
  SHORTEST,  // terminate zip when the first stream is exhausted
  LONGEST    // use nulls for exhausted streams; use up the longest stream.
}

/**
 * Provides a `LazyIterator` that zips together an array, dict, or nested
 * structure of `LazyIterator`s (and perhaps additional constants).
 *
 * The underlying streams must provide elements in a consistent order such
 * that they correspond.
 *
 * Typically, the underlying streams should have the same number of
 * elements. If they do not, the behavior is determined by the
 * `mismatchMode` argument.
 *
 * The nested structure of the `iterators` argument determines the
 * structure of elements in the resulting iterator.
 *
 * Doing this in a concurrency-safe way requires some trickery.  In
 * particular, we want this stream to return the elements from the
 * underlying streams in the correct order according to when next() was
 * called, even if the resulting Promises resolve in a different order.
 *
 * @param iterators: An array or object containing LazyIterators at the
 * leaves.
 * @param mismatchMode: Determines what to do when one underlying iterator
 * is exhausted before the others.  `ZipMismatchMode.FAIL` (the default)
 * causes an error to be thrown in this case.  `ZipMismatchMode.SHORTEST`
 * causes the zipped iterator to terminate with the furst underlying
 * streams, so elements remaining on the longer streams are ignored.
 * `ZipMismatchMode.LONGEST` causes the zipped stream to continue, filling
 * in nulls for the exhausted streams, until all streams are exhausted.
 */
class ZipIterator<O extends tf.TensorContainer> extends LazyIterator<O> {
  private count = 0;
  private currentPromise: Promise<IteratorResult<O>> = null;

  constructor(
      protected readonly iterators: IteratorContainer,
      protected readonly mismatchMode: ZipMismatchMode = ZipMismatchMode.FAIL) {
    super();
  }

  summary() {
    const upstreamSummaries = 'TODO: fill in upstream of zip summaries';
    return `{${upstreamSummaries}} -> Zip`;
  }

  private async nextState(afterState: Promise<IteratorResult<O>>):
      Promise<IteratorResult<O>> {
    // This chaining ensures that the underlying next() are not even called
    // before the previous ones have resolved.
    await afterState;

    // Collect underlying iterator "done" signals as a side effect in
    // getNext()
    let numIterators = 0;
    let iteratorsDone = 0;

    function getNext(container: IteratorContainer): DeepMapAsyncResult {
      if (container instanceof LazyIterator) {
        const result = container.next();
        return {
          value: result.then(x => {
            numIterators++;
            if (x.done) {
              iteratorsDone++;
            }
            return x.value;
          }),
          recurse: false
        };
      } else {
        return {value: null, recurse: true};
      }
    }

    const mapped: O = await deepMapAndAwaitAll(this.iterators, getNext);

    if (numIterators === iteratorsDone) {
      // The streams have all ended.
      return {value: null, done: true};
    }
    if (iteratorsDone > 0) {
      switch (this.mismatchMode) {
        case ZipMismatchMode.FAIL:
          throw new Error(
              'Zipped streams should have the same length. ' +
              `Mismatched at element ${this.count}.`);
        case ZipMismatchMode.SHORTEST:
          return {value: null, done: true};
        case ZipMismatchMode.LONGEST:
        default:
          // Continue.  The exhausted streams already produced value: null.
      }
    }

    this.count++;
    return {value: mapped, done: false};
  }

  async next(): Promise<IteratorResult<O>> {
    this.currentPromise = this.nextState(this.currentPromise);
    return (await this.currentPromise);
  }
}

// Iterators that maintain a ring buffer of pending promises
// ============================================================================

/**
 * A stream that prefetches a given number of items from an upstream source,
 * returning them in FIFO order.
 *
 * Note this prefetches Promises, but makes no guarantees about when those
 * Promises resolve.
 */
export class PrefetchIterator<T> extends LazyIterator<T> {
  protected buffer: RingBuffer<Promise<IteratorResult<T>>>;

  constructor(
      protected upstream: LazyIterator<T>, protected bufferSize: number) {
    super();
    this.buffer = new RingBuffer<Promise<IteratorResult<T>>>(bufferSize);
  }

  summary() {
    return `${this.upstream.summary()} -> Prefetch`;
  }

  /**
   * Refill the prefetch buffer.  Returns only after the buffer is full, or
   * the upstream source is exhausted.
   */
  protected refill() {
    while (!this.buffer.isFull()) {
      const v = this.upstream.next();
      this.buffer.push(v);
    }
  }

  next(): Promise<IteratorResult<T>> {
    this.refill();
    // This shift will never throw an error because the buffer is always
    // full after a refill. If the stream is exhausted, the buffer will be
    // full of Promises that will resolve to the end-of-stream signal.
    return this.buffer.shift();
  }
}

/**
 * A stream that performs a sliding-window random shuffle on an upstream
 * source. This is like a `PrefetchIterator` except that the items are
 * returned in randomized order.  Mixing naturally improves as the buffer
 * size increases.
 */
export class ShuffleIterator<T> extends PrefetchIterator<T> {
  private readonly random: seedrandom.prng;

  // Strict Promise execution order:
  // a next() call may not even begin until the previous one completes.
  private lastRead: Promise<IteratorResult<T>>;

  // Local state that should not be clobbered by out-of-order execution.
  private upstreamExhausted = false;

  constructor(
      protected upstream: LazyIterator<T>, protected windowSize: number,
      seed?: string) {
    super(upstream, windowSize);
    this.random = seedrandom.alea(seed || tf.util.now().toString());
    this.lastRead = Promise.resolve({value: null, done: false});
  }

  async next(): Promise<IteratorResult<T>> {
    // This sets this.lastRead to a new Promise right away, as opposed to
    // saying `await this.lastRead; this.lastRead = this.serialNext();` which
    // would not work because this.nextRead would be updated only after the
    // promise resolves.
    this.lastRead = this.lastRead.then(() => this.serialNext());
    return this.lastRead;
  }

  private randomInt(max: number) {
    return Math.floor(this.random() * max);
  }

  protected chooseIndex(): number {
    return this.randomInt(this.buffer.length());
  }

  async serialNext(): Promise<IteratorResult<T>> {
    // TODO(soergel): consider performance
    if (!this.upstreamExhausted) {
      this.refill();
    }
    while (!this.buffer.isEmpty()) {
      const chosenIndex = this.chooseIndex();
      const result = await this.buffer.shuffleExcise(chosenIndex);
      if (result.done) {
        this.upstreamExhausted = true;
      } else {
        this.refill();
        return result;
      }
    }
    return {value: null, done: true};
  }
}
