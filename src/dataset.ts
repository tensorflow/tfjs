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

import {iteratorFromFunction, iteratorFromZipped, LazyIterator, ZipMismatchMode} from './iterators/lazy_iterator';
import {iteratorFromConcatenated} from './iterators/lazy_iterator';
import {iteratorFromItems} from './iterators/lazy_iterator';
import {DataElement, DatasetContainer} from './types';
import {deepMapAndAwaitAll, DeepMapResult, isIterable, isSubIterable} from './util/deep_map';

// TODO(soergel): consider vectorized operations within the pipeline.

/**
 * Represents a potentially large set of elements.
 *
 * A `Dataset` can be used to represent an input pipeline as a
 * collection of elements (maps from string keys to values) and a "logical
 * plan" of transformations that act on those elements.
 *
 * A `Dataset` provides a stream of unbatched examples, and its transformations
 * are applied one example at a time.  Batching produces a BatchDataset, and so
 * must come last in the pipeline because there are (so far) no batch-enabled
 * transformations.
 */
export abstract class Dataset<T extends DataElement> {
  /*
   * Provide a new stream of elements.  Note this will also start new streams
   * from any underlying `Dataset`s.
   *
   * CAUTION: Any Tensors contained within the elements returned from
   * this stream *must* be manually disposed to avoid a GPU memory leak.
   * The tf.tidy() approach cannot be used in an asynchronous context.
   */
  abstract async iterator(): Promise<LazyIterator<T>>;

  // TODO(soergel): Make Datasets report whether repeated iterator() calls
  // produce the same result (e.g., reading from a file) or different results
  // (e.g., from the webcam).  Currently we don't make this distinction but it
  // could be important for the user to know.
  // abstract isDeterministic(): boolean;

  /**
   * Filters this dataset according to `predicate`.
   *
   * @param predicate A function mapping a dataset element to a boolean or a
   * `Promise` for one.
   *
   * @returns A `Dataset` of elements for which the predicate was true.
   */
  filter(filterer: (value: T) => boolean): Dataset<T> {
    const base = this;
    return datasetFromIteratorFn(async () => {
      return (await base.iterator()).filter(x => tf.tidy(() => filterer(x)));
    });
  }

  /**
   * Maps this dataset through a 1-to-1 transform.
   *
   * @param transform A function mapping a dataset element to a transformed
   *   dataset element.
   *
   * @returns A `Dataset` of transformed elements.
   */
  map<O extends DataElement>(transform: (value: T) => O): Dataset<O> {
    const base = this;
    return datasetFromIteratorFn(async () => {
      return (await base.iterator()).map(x => tf.tidy(() => transform(x)));
    });
  }

  /**
   * Maps this dataset through an async 1-to-1 transform.
   *
   * @param transform A function mapping a dataset element to a `Promise` for a
   *   transformed dataset element.  This transform is responsible for disposing
   *   any intermediate `Tensor`s, i.e. by wrapping its computation in
   *   `tf.tidy()`; that cannot be automated here (as it is in the synchronous
   *   `map()` case).
   *
   * @returns A `Dataset` of transformed elements.
   */
  mapAsync<O extends DataElement>(transform: (value: T) => Promise<O>):
      Dataset<O> {
    const base = this;
    return datasetFromIteratorFn(async () => {
      return (await base.iterator()).mapAsync(transform);
    });
  }

  /**
   * Groups elements into batches and arranges their values in columnar form.
   *
   * It is assumed that each of the incoming dataset elements has the same set
   * of keys.  For each key, the resulting BatchDataset provides a BatchElement
   * collecting all of the incoming values for that key.  Incoming strings are
   * grouped into a string[].  Incoming Tensors are grouped into a new Tensor
   * where the 0'th axis is the batch dimension.  These columnar representations
   * for each key can be zipped together to reconstruct the original
   * dataset elements.
   *
   * @param batchSize The number of elements desired per batch.
   * @param smallLastBatch Whether to emit the final batch when it has fewer
   *   than batchSize elements. Default true.
   * @returns A `BatchDataset`, from which a stream of batches can be obtained.
   */
  batch(batchSize: number, smallLastBatch = true): Dataset<DataElement> {
    const base = this;
    return datasetFromIteratorFn(async () => {
      return (await base.iterator())
          .columnMajorBatch(batchSize, smallLastBatch, deepBatchConcat);
    });
  }

  /**
   * Concatenates this `Dataset` with another.
   *
   * @param dataset A `Dataset` to be concatenated onto this one.
   * @returns A `Dataset`.
   */
  concatenate(dataset: Dataset<T>): Dataset<T> {
    const base = this;
    return datasetFromIteratorFn(
        async () =>
            (await base.iterator()).concatenate(await dataset.iterator()));
  }

  /**
   * Repeats this dataset `count` times.
   *
   * NOTE: If this dataset is a function of global state (e.g. a random number
   * generator), then different repetitions may produce different elements.
   *
   * @param count: (Optional.) An integer, representing the number of times
   *   the dataset should be repeated. The default behavior (if `count` is
   *   `undefined` or negative) is for the dataset be repeated indefinitely.
   * @returns A `Dataset`.
   */
  repeat(count?: number): Dataset<T> {
    const base = this;
    return datasetFromIteratorFn(async () => {
      const iteratorIterator = iteratorFromFunction(
          async () => ({value: await base.iterator(), done: false}));
      return iteratorFromConcatenated(iteratorIterator.take(count));
    });
  }

  /**
   * Creates a `Dataset` with at most `count` elements from this dataset.
   *
   * @param count: The number of elements of this dataset that should be taken
   *   to form the new dataset.  If `count` is `undefined` or negative, or if
   *   `count` is greater than the size of this dataset, the new dataset will
   *   contain all elements of this dataset.
   * @returns A `Dataset`.
   */
  take(count: number): Dataset<T> {
    const base = this;
    return datasetFromIteratorFn(
        async () => (await base.iterator()).take(count));
  }

  /**
   * Creates a `Dataset` that skips `count` elements from this dataset.
   *
   * @param count: The number of elements of this dataset that should be skipped
   *   to form the new dataset.  If `count` is greater than the size of this
   *   dataset, the new dataset will contain no elements.  If `count`
   *   is `undefined` or negative, skips the entire dataset.
   *
   * @returns A `Dataset`.
   */
  skip(count: number): Dataset<T> {
    const base = this;
    return datasetFromIteratorFn(
        async () => (await base.iterator()).skip(count));
  }

  // TODO(soergel): deep sharded shuffle, where supported

  /**
   * Randomly shuffles the elements of this dataset.
   *
   * @param bufferSize: An integer specifying the number of elements from this
   *   dataset from which the new dataset will sample.
   * @param seed: (Optional.) An integer specifying the random seed that will
   *   be used to create the distribution.
   * @param reshuffleEachIteration: (Optional.) A boolean, which if true
   *   indicates that the dataset should be pseudorandomly reshuffled each time
   *   it is iterated over. (Defaults to `true`.)
   * @returns A `Dataset`.
   */
  shuffle(bufferSize: number, seed?: string, reshuffleEachIteration = true):
      Dataset<T> {
    const base = this;
    const random = seedrandom.alea(seed || performance.now().toString());
    return datasetFromIteratorFn(async () => {
      let seed2 = random.int32();
      if (reshuffleEachIteration) {
        seed2 += random.int32();
      }
      return (await base.iterator()).shuffle(bufferSize, seed2.toString());
    });
  }

  /**
   *  Creates a `Dataset` that prefetches elements from this Dataset.
   *
   * @param bufferSize: An integer specifying the number of elements to be
   *   prefetched.
   * @returns A `Dataset`.
   */
  prefetch(bufferSize: number): Dataset<T> {
    const base = this;
    return datasetFromIteratorFn(
        async () => (await base.iterator()).prefetch(bufferSize));
  }

  /**
   * Collect all elements of this dataset into an array.
   * Obviously this will succeed only for small datasets that fit in memory.
   * Useful for testing.
   *
   * @returns A Promise for an array of elements, which will resolve
   *   when a new stream has been obtained and fully consumed.
   */
  async collectAll() {
    return (await this.iterator()).collect();
  }

  /**
   * Apply a function to every element of the dataset.
   *
   * After the function is applied to a dataset element, any Tensors contained
   * within that element are disposed.
   *
   * @param f A function to apply to each dataset element.
   * @returns A `Promise` that resolves after all elements have been processed.
   */
  async forEach(f: (input: T) => void): Promise<void> {
    return (await this.iterator()).forEach(f);
  }

  /* TODO(soergel): for parity with tf.data:
  Dataset.flat_map()
  Dataset.dense_to_sparse_batch()
  Dataset.group_by_window()
  Dataset.padded_batch()
  */
}

/**
 * Create a `Dataset` defined by a provided iterator() function.
 */
export function datasetFromIteratorFn<T extends DataElement>(
    iteratorFn: () => Promise<LazyIterator<T>>): Dataset<T> {
  return new class extends Dataset<T> {
    /*
     * Provide a new stream of elements.  Note this will also start new streams
     * from any underlying `Dataset`s.
     */
    async iterator(): Promise<LazyIterator<T>> {
      return iteratorFn();
    }
  }
  ();
}

/**
 * Create a `Dataset` from an array of elements.
 */
export function datasetFromElements<T extends DataElement>(items: T[]):
    Dataset<T> {
  return datasetFromIteratorFn(async () => iteratorFromItems(items));
}

/**
 * Create a `Dataset` by zipping together an array, dict, or nested
 * structure of `Dataset`s (and perhaps additional constants).
 * The underlying datasets must provide elements in a consistent order such that
 * they correspond.
 *
 * The number of elements in the resulting dataset is the same as the size of
 * the smallest dataset in `datasets`.
 *
 * The nested structure of the `datasets` argument determines the
 * structure of elements in the resulting iterator.
 *
 * Note this means that, given an array of two datasets that produce dict
 * elements, the result is a dataset that produces elements that are arrays
 * of two dicts:
 *
 * const ds1 : Dataset = ...;  // produces elements like {a: ...}
 * const ds1 : Dataset = ...;  // produces elements like {b: ...}
 * const ds3 = zip([ds1, ds2]);  // produces elements like [{a: ...}, {b: ...}]
 *
 * If the goal is to merge the dicts in order to produce elements like
 * {a: ..., b: ...}, this requires a second step such as:
 *
 * const ds4 = ds3.map(x=>{a: x[0].a, b: x[1].b});
 */
export function zip<O extends DataElement>(datasets: DatasetContainer):
    Dataset<O> {
  // manually type-check the argument for JS users
  if (!isIterable(datasets)) {
    throw new Error('The argument to zip() must be an object or array.');
  }
  return datasetFromIteratorFn<O>(async () => {
    const streams = await deepMapAndAwaitAll(datasets, d => {
      if (d instanceof Dataset) {
        return {value: d.iterator(), recurse: false};
      } else if (isIterable(d)) {
        return {value: null, recurse: true};
      } else {
        throw new Error(
            'Leaves of the structure passed to zip() must be Datasets, ' +
            'not primitives.');
      }
    });
    return iteratorFromZipped<O>(streams, ZipMismatchMode.SHORTEST);
  });
}

/**
 * A zip function for use with deepZip, passed via the columnMajorBatch call.
 *
 * Accepts an array of identically-structured nested elements and either batches
 * them (if they are primitives, numeric arrays, or Tensors) or requests
 * recursion (if not).
 */
// tslint:disable-next-line:no-any
function deepBatchConcat(x: any[]): DeepMapResult {
  if (x === null) {
    return null;
  }
  // TODO(soergel): validate array type?
  // TODO(soergel): performance: avoid testing each item twice
  if (isIterable(x[0]) && isSubIterable(x[0])) {
    return {value: null, recurse: true};
  } else if (typeof (x[0]) === 'string') {
    // TODO(soergel): clean up the string special case when Tensor supports it.
    return {value: x, recurse: false};
  } else {
    return {value: batchConcat(x), recurse: false};
  }
}

/**
 * Assembles a list of same-shaped numbers, number arrays, or Tensors
 * into a single new Tensor where axis 0 is the batch dimension.
 */
function batchConcat(arrays: Array<number|number[]|tf.Tensor>): tf.Tensor {
  // Should we use GPU-enabled concat ops in deeplearn's math.ts?
  // Probably not; the GPU roundtrip is not worth it for a trivial
  // operation.
  const [elementShape, ] = shapeAndValues(arrays[0]);
  const batchShape = [arrays.length].concat(elementShape);
  const resultVals = new Float32Array(batchShape.reduce((x, y) => x * y));

  let offset = 0;
  for (const a of arrays) {
    const [aShape, aVals] = shapeAndValues(a);
    if (!tf.util.arraysEqual(aShape, elementShape)) {
      throw new Error('Elements must have the same shape to be batched');
    }
    resultVals.set(aVals, offset);
    offset += aVals.length;
  }
  return tf.Tensor.make(batchShape, {values: resultVals});
}

/**
 * Extracts the shape and values from the argument, whether array or Tensor.
 *
 * If the argument is a Tensor, this performs a 'dataSync()' to obtain the
 * values as a typed Array.
 *
 * @returns a tuple where the first element is a number[] describing the shape
 * and the second is a number[] or a TypedArray containing the values.
 */
function shapeAndValues(array: number|number[]|tf.Tensor):
    [number[], number[]|Float32Array|Int32Array|Uint8Array] {
  if (array instanceof tf.Tensor) {
    return [array.shape, array.dataSync()];
  } else if (Array.isArray(array)) {
    return [[array.length], array];
  } else {
    return [[], [array]];
  }
}
