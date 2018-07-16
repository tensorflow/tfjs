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

// tslint:disable:max-line-length
import * as tf from '@tensorflow/tfjs-core';
import {getTensorsInContainer, isTensorInList} from '@tensorflow/tfjs-core/dist/tensor_util';
import * as seedrandom from 'seedrandom';
// tslint:enable:max-line-length

import {DataElement, IteratorContainer} from '../types';
import {deepMapAndAwaitAll, DeepMapAsyncResult} from '../util/deep_map';
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
 */
export function iteratorFromConcatenated<T>(
    baseIterators: LazyIterator<LazyIterator<T>>): LazyIterator<T> {
  return ChainedIterator.create(baseIterators);
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
 */
export function iteratorFromConcatenatedFunction<T>(
    iteratorFunc: () => IteratorResult<LazyIterator<T>>,
    count: number): LazyIterator<T> {
  return iteratorFromConcatenated(
      iteratorFromFunction(iteratorFunc).take(count));
}

/**
 * Create a `LazyIterator` by zipping together an array, dict, or nested
 * structure of `LazyIterator`s (and perhaps additional constants).
 *
 * The underlying streams must provide elements in a consistent order such that
 * they correspond.
 *
 * Typically, the underlying streams should have the same number of elements.
 * If they do not, the behavior is determined by the `mismatchMode` argument.
 *
 * The nested structure of the `iterators` argument determines the
 * structure of elements in the resulting iterator.
 *
 * @param iterators: An array or object containing LazyIterators at the leaves.
 * @param mismatchMode: Determines what to do when one underlying iterator is
 *   exhausted before the others.  `ZipMismatchMode.FAIL` (the default) causes
 *   an error to be thrown in this case.  `ZipMismatchMode.SHORTEST` causes the
 *   zipped iterator to terminate with the furst underlying streams, so elements
 *   remaining on the longer streams are ignored.  `ZipMismatchMode.LONGEST`
 *   causes the zipped stream to continue, filling in nulls for the exhausted
 *   streams, until all streams are exhausted.
 */
export function iteratorFromZipped(
    iterators: IteratorContainer,
    mismatchMode: ZipMismatchMode =
        ZipMismatchMode.FAIL): LazyIterator<DataElement> {
  return new ZipIterator(iterators, mismatchMode);
}

export class IteratorProperties {
  // Is each returned item an independent unit (such as an example or a batch),
  // as opposed to a stream segment (like a chunk of a file)?
  independent: boolean;

  // Is the iteration order meaningful?
  ordered: boolean;

  // How many initial dimensions of contained Tensors are batch dimensions.
  // i.e. 0 means we have independent examples, 1 means we have normal batches,
  // 2 means we have batches of batches.
  batchDimensions: number;

  columnarBatchDimensions: number;
}

/**
 * An asynchronous iterator, providing lazy access to a potentially unbounded
 * stream of elements.
 */
export abstract class LazyIterator<T> {
  // This class implements AsyncIterator<T>, but we have not yet set the
  // TypeScript --downlevelIteration flag to enable that.
  properties: IteratorProperties;

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
  async collectRemaining(): Promise<T[]> {
    const result: T[] = [];
    let x = await this.next();
    while (!x.done) {
      result.push(x.value);
      x = await this.next();
    }
    return result;
  }

  /**
   * Draw items from the stream until it is exhausted.
   *
   * This can be useful when the stream has side effects but no output.  In
   * that case, calling this function guarantees that the stream will be fully
   * processed.
   */
  async resolveFully(): Promise<void> {
    let x = await this.next();
    while (!x.done) {
      x = await this.next();
    }
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
   * @param predicate A function mapping a stream element to a transformed
   *   element.
   *
   * @returns A `LazyIterator` of transformed elements.
   */
  map<O>(transform: (value: T) => O): LazyIterator<O> {
    return new MapIterator(this, transform);
  }

  /**
   * Maps this stream through a 1-to-many transform.
   *
   * @param predicate A function mapping a stream element to an array of
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
  async forEach(f: (value: T) => void): Promise<void> {
    return this.map(f).resolveFully();
  }

  /**
   * Groups elements into batches.
   *
   * @param batchSize The number of elements desired per batch.
   * @param smallLastBatch Whether to emit the final batch when it has fewer
   *   than batchSize elements. Default true.
   * @returns A `LazyIterator` of batches of elements, represented as arrays
   *   of the original element type.
   */
  batch(batchSize: number, smallLastBatch = true): LazyIterator<T[]> {
    return new BatchIterator(this, batchSize, smallLastBatch);
  }

  /**
   * Concatenate this `LazyIterator` with another.
   *
   * @param iterator A `LazyIterator` to be concatenated onto this one.
   * @returns A `LazyIterator`.
   */
  concatenate(iterator: LazyIterator<T>): LazyIterator<T> {
    return ChainedIterator.create(iteratorFromItems([this, iterator]));
  }

  /**
   * Limits this stream to return at most `count` items.
   *
   * @param count The maximum number of items to provide from the stream.  If a
   *   negative or undefined value is given, the entire stream is returned
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
   * @param count The number of items to skip.  If a negative or undefined value
   *   is given, the entire stream is returned unaltered.
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
   * @param bufferSize: An integer specifying the number of elements from this
   *   stream from which the new stream will sample.
   * @param seed: (Optional.) An integer specifying the random seed that will
   *   be used to create the distribution.
   */
  shuffle(windowSize: number, seed?: string): LazyIterator<T> {
    return new ShuffleIterator(this, windowSize, seed);
  }
}

// ============================================================================
// The following private classes serve to implement the chainable methods
// on LazyIterator.  Unfortunately they can't be placed in separate files, due
// to resulting trouble with circular imports.
// ============================================================================

// Iterators that just extend LazyIterator directly
// ============================================================================

class ArrayIterator<T> extends LazyIterator<T> {
  private trav = 0;
  constructor(protected items: T[]) {
    super();
  }

  async next(): Promise<IteratorResult<T>> {
    if (this.trav >= this.items.length) {
      return {value: null, done: true};
    }
    const result = this.items[this.trav];
    this.trav++;
    return {value: result, done: false};
  }
}

class FunctionCallIterator<T> extends LazyIterator<T> {
  constructor(
      protected nextFn: () => IteratorResult<T>| Promise<IteratorResult<T>>) {
    super();
  }

  async next(): Promise<IteratorResult<T>> {
    try {
      return this.nextFn();
    } catch (e) {
      // Modify the error message but leave the stack trace intact
      e.message =
          'Error thrown while iterating through a dataset: ' + e.message;
      throw e;
    }
  }
}

class SkipIterator<T> extends LazyIterator<T> {
  count = 0;
  constructor(protected upstream: LazyIterator<T>, protected maxCount: number) {
    super();
  }

  async next(): Promise<IteratorResult<T>> {
    // TODO(soergel): consider tradeoffs of reading in parallel, eg. collecting
    // next() promises in an Array and then waiting for Promise.all() of those.
    // Benefit: pseudo-parallel execution.  Drawback: maybe delayed GC.
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

  async next(): Promise<IteratorResult<T>> {
    if (this.count++ >= this.maxCount) {
      return {value: null, done: true};
    }
    return this.upstream.next();
  }
}

class BatchIterator<T> extends LazyIterator<T[]> {
  constructor(
      protected upstream: LazyIterator<T>, protected batchSize: number,
      protected enableSmallLastBatch = true) {
    super();
  }
  async next(): Promise<IteratorResult<T[]>> {
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
  constructor(
      protected upstream: LazyIterator<T>,
      protected predicate: (value: T) => boolean) {
    super();
  }
  async next(): Promise<IteratorResult<T>> {
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
  async next(): Promise<IteratorResult<O>> {
    const item = await this.upstream.next();
    if (item.done) {
      return {value: null, done: true};
    }
    const inputTensors = getTensorsInContainer(item.value as {});
    // Careful: the transform may mutate the item in place.
    // that's why we have to remember the input Tensors above, and then
    // below
    // dispose only those that were not passed through to the output.
    // Note too that the transform function is responsible for tidying
    // any
    // intermediate Tensors.  Here we are concerned only about the
    // inputs.
    const mapped = this.transform(item.value);
    const outputTensors = getTensorsInContainer(mapped as {});
    // TODO(soergel) faster intersection
    // TODO(soergel) move to tf.disposeExcept(in, out)?
    for (const t of inputTensors) {
      if (!isTensorInList(t, outputTensors)) {
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
 * may trigger a call to the underlying stream, which will produce many mapped
 * elements of this stream-- of which we need to return only one, so we have to
 * queue the rest.
 */
export abstract class QueueIterator<T> extends LazyIterator<T> {
  protected outputQueue: RingBuffer<T>;

  constructor() {
    super();
    this.outputQueue = new GrowingRingBuffer<T>();
  }
  /**
   * Read one or more chunks from upstream and process them, possibly reading or
   * writing a carryover, and adding processed items to the output queue.  Note
   * it's possible that no items are added to the queue on a given
   * pump() call, even if the upstream stream is not closed (e.g., because items
   * are filtered).
   *
   * @return `true` if any action was taken, i.e. fetching items from the
   *   upstream source OR adding items to the output queue.  `false` if the
   *   upstream source is exhausted AND nothing was added to the queue (i.e.,
   *   any remaining carryover).
   */
  protected abstract async pump(): Promise<boolean>;

  async next(): Promise<IteratorResult<T>> {
    // Fetch so that the queue contains at least one item if possible.
    // If the upstream source is exhausted, AND there are no items left in the
    // output queue, then this stream is also exhausted.
    while (this.outputQueue.length() === 0) {
      // TODO(soergel): consider parallel reads.
      if (!await this.pump()) {
        return {value: null, done: true};
      }
    }
    return {value: this.outputQueue.shift(), done: false};
  }
}
class FlatmapIterator<I, O> extends QueueIterator<O> {
  constructor(
      protected upstream: LazyIterator<I>,
      protected transform: (value: I) => O[]) {
    super();
  }

  async pump(): Promise<boolean> {
    const item = await this.upstream.next();
    if (item.done) {
      return false;
    }
    const inputTensors = getTensorsInContainer(item.value as {});
    // Careful: the transform may mutate the item in place.
    // that's why we have to remember the input Tensors above, and then below
    // dispose only those that were not passed through to the output.
    // Note too that the transform function is responsible for tidying any
    // intermediate Tensors.  Here we are concerned only about the inputs.
    const mappedArray = this.transform(item.value);
    const outputTensors = getTensorsInContainer(mappedArray as {});
    this.outputQueue.pushAll(mappedArray);

    // TODO(soergel) faster intersection, and deduplicate outputTensors
    // TODO(soergel) move to tf.disposeExcept(in, out)?
    for (const t of inputTensors) {
      if (!isTensorInList(t, outputTensors)) {
        t.dispose();
      }
    }

    return true;
  }
}
/**
 * Provides a `LazyIterator` that concatenates a stream of underlying streams.
 *
 * Doing this in a concurrency-safe way requires some trickery.  In particular,
 * we want this stream to return the elements from the underlying streams in
 * the correct order according to when next() was called, even if the resulting
 * Promises resolve in a different order.
 */
export class ChainedIterator<T> extends LazyIterator<T> {
  private iterator: LazyIterator<T> = null;
  private moreIterators: LazyIterator<LazyIterator<T>>;
  private lastRead: Promise<IteratorResult<T>> = null;

  static create<T>(iterators: LazyIterator<LazyIterator<T>>):
      ChainedIterator<T> {
    const c = new ChainedIterator<T>();
    c.moreIterators = iterators;
    return c;
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
    // TODO(smilkov): Rework logic to allow parallel reads.
    await lastRead;
    if (this.iterator == null) {
      const iteratorResult = await this.moreIterators.next();
      if (iteratorResult.done) {
        // No more streams to stream from.
        return {value: null, done: true};
      }
      this.iterator = iteratorResult.value;
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
 * The underlying streams must provide elements in a consistent order such that
 * they correspond.
 *
 * Typically, the underlying streams should have the same number of elements.
 * If they do not, the behavior is determined by the `mismatchMode` argument.
 *
 * The nested structure of the `iterators` argument determines the
 * structure of elements in the resulting iterator.
 *
 * Doing this in a concurrency-safe way requires some trickery.  In particular,
 * we want this stream to return the elements from the underlying streams in
 * the correct order according to when next() was called, even if the resulting
 * Promises resolve in a different order.
 *
 * @param iterators: An array or object containing LazyIterators at the leaves.
 * @param mismatchMode: Determines what to do when one underlying iterator is
 *   exhausted before the others.  `ZipMismatchMode.FAIL` (the default) causes
 *   an error to be thrown in this case.  `ZipMismatchMode.SHORTEST` causes the
 *   zipped iterator to terminate with the furst underlying streams, so elements
 *   remaining on the longer streams are ignored.  `ZipMismatchMode.LONGEST`
 *   causes the zipped stream to continue, filling in nulls for the exhausted
 *   streams, until all streams are exhausted.
 */
class ZipIterator extends LazyIterator<DataElement> {
  private count = 0;
  private currentPromise: Promise<IteratorResult<DataElement>> = null;

  constructor(
      protected readonly iterators: IteratorContainer,
      protected readonly mismatchMode: ZipMismatchMode = ZipMismatchMode.FAIL) {
    super();
  }

  private async nextState(afterState: Promise<IteratorResult<DataElement>>):
      Promise<IteratorResult<DataElement>> {
    // This chaining ensures that the underlying next() are not even called
    // before the previous ones have resolved.
    await afterState;

    // Collect underlying iterator "done" signals as a side effect in getNext()
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

    const mapped = await deepMapAndAwaitAll(this.iterators, getNext);

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

  async next(): Promise<IteratorResult<DataElement>> {
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

  total = 0;

  constructor(
      protected upstream: LazyIterator<T>, protected bufferSize: number) {
    super();
    this.buffer = new RingBuffer<Promise<IteratorResult<T>>>(bufferSize);
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
    // This shift will never throw an error because the buffer is always full
    // after a refill. If the stream is exhausted, the buffer will be full of
    // Promises that will resolve to the end-of-stream signal.
    return this.buffer.shift();
  }
}

/**
 * A stream that performs a sliding-window random shuffle on an upstream
 * source. This is like a `PrefetchIterator` except that the items are returned
 * in randomized order.  Mixing naturally improves as the buffer size
 * increases.
 */
export class ShuffleIterator<T> extends PrefetchIterator<T> {
  private random: seedrandom.prng;
  private upstreamExhausted = false;

  constructor(
      protected upstream: LazyIterator<T>, protected windowSize: number,
      seed?: string) {
    super(upstream, windowSize);
    this.random = seedrandom.alea(seed || performance.now().toString());
  }

  private randomInt(max: number) {
    return Math.floor(this.random() * max);
  }

  protected chooseIndex(): number {
    return this.randomInt(this.buffer.length());
  }

  async next(): Promise<IteratorResult<T>> {
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
