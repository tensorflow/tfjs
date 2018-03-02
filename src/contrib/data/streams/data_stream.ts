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

import * as seedrandom from 'seedrandom';

import {dispose} from '../../../globals';
import {extractTensorsFromAny, isTensorInList} from '../../../util';
import {GrowingRingBuffer} from '../util/growing_ring_buffer';
import {RingBuffer} from '../util/ring_buffer';

// Here we implement a simple asynchronous iterator.
// This lets us avoid using either third-party stream libraries or
// recent TypeScript language support requiring polyfills.
// Note we return Promise<T>, not Promise<IteratorResult<T>>, so this might
// require slight retrofitting in the future if we want to use the ES6 features.

/**
 * Create a `DataStream` from an array of items.
 */
export function streamFromItems<T>(items: T[]): DataStream<T> {
  return new ArrayStream(items);
}

/**
 * Create a `DataStream` of incrementing integers.
 */
export function streamFromIncrementing(start: number): DataStream<number> {
  let i = start;
  return streamFromFunction(() => i++);
}

/**
 * Create a `DataStream` from a function.
 */
export function streamFromFunction<T>(func: () => T | Promise<T>):
    DataStream<T> {
  return new FunctionCallStream(func);
}

/**
 * Create a `DataStream` by concatenating underlying streams, which are
 * themselves provided as a stream.
 *
 * This can also be thought of as a "stream flatten" operation.
 *
 * @param baseStreams A stream of streams to be concatenated.
 */
export function streamFromConcatenated<T>(
    baseStreams: DataStream<DataStream<T>>): DataStream<T> {
  return ChainedStream.create(baseStreams);
}

/**
 * Create a `DataStream` by concatenating streams produced by calling a
 * stream-generating function a given number of times.
 *
 * Since a `DataStream` is read-once, it cannot be repeated, but this
 * function can be used to achieve a similar effect:
 *
 *   DataStream.ofConcatenatedFunction(() => new MyStream(), 6);
 *
 * @param streamFunc: A function that produces a new stream on each call.
 * @param count: The number of times to call the function.
 */
export function streamFromConcatenatedFunction<T>(
    streamFunc: () => DataStream<T>, count: number): DataStream<T> {
  return streamFromConcatenated(streamFromFunction(streamFunc).take(count));
}

/**
 * An asynchronous iterator, providing lazy access to a potentially unbounded
 * stream of elements.
 */
export abstract class DataStream<T> {
  /**
   * Returns a `Promise` for the next element in the stream.
   *
   * Calling next() on a closed stream returns `undefined`.
   */
  abstract async next(): Promise<T>;

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
    while (x != null) {
      result.push(x);
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
    while (x != null) {
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
   * @returns A `DataStream` of elements for which the predicate was true.
   */
  filter(predicate: (value: T) => boolean): DataStream<T> {
    return new FilterStream(this, predicate);
  }

  /**
   * Maps this stream through a 1-to-1 transform.
   *
   * @param predicate A function mapping a stream element to a transformed
   *   element.
   *
   * @returns A `DataStream` of transformed elements.
   */
  map<O>(transform: (value: T) => O): DataStream<O> {
    return new MapStream(this, transform);
  }

  /**
   * Apply a function to every element of the stream.
   *
   * @param f A function to apply to each stream element.
   */
  async forEach(f: (value: T) => {}): Promise<void> {
    return this.map(f).resolveFully();
  }

  /**
   * Groups elements into batches.
   *
   * @param batchSize The number of elements desired per batch.
   * @param smallLastBatch Whether to emit the final batch when it has fewer
   *   than batchSize elements. Default true.
   * @returns A `DataStream` of batches of elements, represented as arrays
   *   of the original element type.
   */
  batch(batchSize: number, smallLastBatch = true): DataStream<T[]> {
    return new BatchStream(this, batchSize, smallLastBatch);
  }

  /**
   * Concatenate this `DataStream` with another.
   *
   * @param stream A `DataStream` to be concatenated onto this one.
   * @returns A `DataStream`.
   */
  concatenate(stream: DataStream<T>): DataStream<T> {
    return ChainedStream.create(streamFromItems([this, stream]));
  }

  /**
   * Limits this stream to return at most `count` items.
   *
   * @param count The maximum number of items to provide from the stream.  If a
   *   negative or undefined value is given, the entire stream is returned
   *   unaltered.
   */
  take(count: number): DataStream<T> {
    if (count < 0 || count == null) {
      return this;
    }
    return new TakeStream(this, count);
  }

  /**
   * Skips the first `count` items in this stream.
   *
   * @param count The number of items to skip.  If a negative or undefined value
   *   is given, the entire stream is returned unaltered.
   */
  skip(count: number): DataStream<T> {
    if (count < 0 || count == null) {
      return this;
    }
    return new SkipStream(this, count);
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
  prefetch(bufferSize: number): DataStream<T> {
    return new PrefetchStream(this, bufferSize);
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
  shuffle(windowSize: number, seed?: string): DataStream<T> {
    return new ShuffleStream(this, windowSize, seed);
  }
}

// ============================================================================
// The following private classes serve to implement the chainable methods
// on DataStream.  Unfortunately they can't be placed in separate files, due to
// resulting trouble with circular imports.
// ============================================================================

// Streams that just extend DataStream directly
// ============================================================================

class ArrayStream<T> extends DataStream<T> {
  private trav = 0;
  constructor(protected items: T[]) {
    super();
  }

  async next(): Promise<T> {
    if (this.trav >= this.items.length) {
      return undefined;
    }
    const result = this.items[this.trav];
    this.trav++;
    return result;
  }
}

class FunctionCallStream<T> extends DataStream<T> {
  constructor(protected nextFn: () => T | Promise<T>) {
    super();
  }

  async next(): Promise<T> {
    return this.nextFn();
  }
}

class SkipStream<T> extends DataStream<T> {
  count = 0;
  constructor(protected upstream: DataStream<T>, protected maxCount: number) {
    super();
  }

  async next(): Promise<T> {
    while (this.count++ < this.maxCount) {
      const skipped = await this.upstream.next();
      // short-circuit if upstream is already empty
      if (skipped == null) {
        return undefined;
      }
      dispose(skipped);
    }
    return this.upstream.next();
  }
}

class TakeStream<T> extends DataStream<T> {
  count = 0;
  constructor(protected upstream: DataStream<T>, protected maxCount: number) {
    super();
  }

  async next(): Promise<T> {
    if (this.count++ >= this.maxCount) {
      return undefined;
    }
    return this.upstream.next();
  }
}

// Streams that maintain a queue of pending items
// ============================================================================

/**
 * A base class for transforming streams that operate by maintaining an
 * output queue of elements that are ready to return via next().  This is
 * commonly required when the transformation is not 1-to-1, so a variable number
 * of calls to the underlying stream may be needed to provide each element of
 * this stream.
 */
export abstract class QueueStream<T> extends DataStream<T> {
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

  async next(): Promise<T> {
    // Fetch so that the queue contains at least one item if possible.
    // If the upstream source is exhausted, AND there are no items left in the
    // output queue, then this stream is also exhausted.
    while (this.outputQueue.length() === 0) {
      if (!await this.pump()) {
        return undefined;
      }
    }
    return this.outputQueue.shift();
  }
}

class BatchStream<T> extends QueueStream<T[]> {
  constructor(
      protected upstream: DataStream<T>, protected batchSize: number,
      protected enableSmallLastBatch = true) {
    super();
  }

  private currentBatch: T[] = [];

  async pump(): Promise<boolean> {
    const item = await this.upstream.next();
    if (item == null) {
      if (this.enableSmallLastBatch && this.currentBatch.length > 0) {
        this.outputQueue.push(this.currentBatch);
        this.currentBatch = [];

        // Pretend that the pump succeeded in order to emit the small last
        // batch. The next pump() call will actually fail.
        return true;
      }
      return false;
    }

    this.currentBatch.push(item);
    if (this.currentBatch.length === this.batchSize) {
      this.outputQueue.push(this.currentBatch);
      this.currentBatch = [];
    }
    return true;
  }
}

class FilterStream<T> extends QueueStream<T> {
  constructor(
      protected upstream: DataStream<T>,
      protected predicate: (value: T) => boolean) {
    super();
  }

  async pump() {
    const item = await this.upstream.next();
    if (item == null) {
      return false;
    }
    if (this.predicate(item)) {
      this.outputQueue.push(item);
    } else {
      dispose(item);
    }
    return true;
  }
}

class MapStream<I, O> extends QueueStream<O> {
  constructor(
      protected upstream: DataStream<I>, protected transform: (value: I) => O) {
    super();
  }

  async pump() {
    const item = await this.upstream.next();
    if (item == null) {
      return false;
    }
    const inputTensors = extractTensorsFromAny(item);
    // Careful: the transform may mutate the item in place.
    // that's why we have to remember the input Tensors above, and then below
    // dispose only those that were not passed through to the output.
    // Note too that the transform function is responsible for tidying any
    // intermediate Tensors.  Here we are concerned only about the inputs.
    const mapped = this.transform(item);

    const outputTensors = extractTensorsFromAny(mapped);

    // TODO(soergel) faster intersection
    // TODO(soergel) move to dl.disposeExcept(in, out)?
    for (const t of inputTensors) {
      if (!isTensorInList(t, outputTensors)) {
        t.dispose();
      }
    }

    this.outputQueue.push(mapped);
    return true;
  }
}

/**
 * Provides a `DataStream` that concatenates a stream of underlying streams.
 *
 * Doing this in a concurrency-safe way requires some trickery.  In particular,
 * we want this stream to return the elements from the underlying streams in
 * the correct order according to when next() was called, even if the resulting
 * Promises resolve in a different order.
 */
export class ChainedStream<T> extends DataStream<T> {
  private stream: DataStream<T> = null;
  private moreStreams: DataStream<DataStream<T>>;
  private lastRead: Promise<T> = null;

  static create<T>(streams: DataStream<DataStream<T>>): ChainedStream<T> {
    const c = new ChainedStream<T>();
    c.moreStreams = streams;
    return c;
  }

  async next(): Promise<T> {
    this.lastRead = this.readFromChain(this.lastRead);
    return this.lastRead;
  }

  private async readFromChain(lastRead: Promise<T>): Promise<T> {
    // Must await on the previous read since the previous read may have advanced
    // the stream of streams, from which we need to read.
    // This is unfortunate since we can't parallelize reads. Which means
    // prefetching of chained streams is a no-op.
    // TODO(smilkov): Rework logic to allow parallel reads.
    await lastRead;
    if (this.stream == null) {
      this.stream = await this.moreStreams.next();
      if (this.stream == null) {
        // No more streams to stream from.
        return null;
      }
    }
    const item = await this.stream.next();
    if (item == null) {
      this.stream = null;
      return this.readFromChain(lastRead);
    }
    return item;
  }
}

// Streams that maintain a ring buffer of pending promises
// ============================================================================

/**
 * A stream that prefetches a given number of items from an upstream source,
 * returning them in FIFO order.
 *
 * Note this prefetches Promises, but makes no guarantees about when those
 * Promises resolve.
 */
export class PrefetchStream<T> extends DataStream<T> {
  protected buffer: RingBuffer<Promise<T>>;

  total = 0;

  constructor(protected upstream: DataStream<T>, protected bufferSize: number) {
    super();
    this.buffer = new RingBuffer<Promise<T>>(bufferSize);
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

  next(): Promise<T> {
    this.refill();
    // This shift will never throw an error because the buffer is always full
    // after a refill. If the stream is exhausted, the buffer will be full of
    // Promises that will resolve to the end-of-stream signal.
    return this.buffer.shift();
  }
}

/**
 * A stream that performs a sliding-window random shuffle on an upstream
 * source. This is like a `PrefetchStream` except that the items are returned
 * in randomized order.  Mixing naturally improves as the buffer size
 * increases.
 */
export class ShuffleStream<T> extends PrefetchStream<T> {
  private random: seedrandom.prng;
  private upstreamExhausted = false;

  constructor(
      protected upstream: DataStream<T>, protected windowSize: number,
      seed?: string) {
    super(upstream, windowSize);
    this.random = seedrandom(seed);
  }

  private randomInt(max: number) {
    return Math.floor(this.random() * max);
  }

  protected chooseIndex(): number {
    return this.randomInt(this.buffer.length());
  }

  async next(): Promise<T> {
    // TODO(soergel): consider performance
    if (!this.upstreamExhausted) {
      this.refill();
    }
    while (!this.buffer.isEmpty()) {
      const chosenIndex = this.chooseIndex();
      const result = await this.buffer.shuffleExcise(chosenIndex);
      if (result == null) {
        this.upstreamExhausted = true;
      } else {
        this.refill();
        return result;
      }
    }
    return undefined;
  }
}
