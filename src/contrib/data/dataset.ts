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

import {BatchDataset} from './batch_dataset';
import {DataStream} from './streams/data_stream';
import {streamFromConcatenated} from './streams/data_stream';
import {streamFromFunction} from './streams/data_stream';
import {streamFromItems} from './streams/data_stream';
import {DatasetElement} from './types';

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
export abstract class Dataset {
  /*
   * Provide a new stream of elements.  Note this will also start new streams
   * from any underlying `Dataset`s.
   */
  abstract async getStream(): Promise<DataStream<DatasetElement>>;

  /**
   * Filters this dataset according to `predicate`.
   *
   * @param predicate A function mapping a `DatasetElement` to a boolean or a
   * `Promise` for one.
   *
   * @returns A `Dataset` of elements for which the predicate was true.
   */
  filter(filterer: (value: DatasetElement) => boolean | Promise<boolean>):
      Dataset {
    const base = this;
    return datasetFromStreamFn(async () => {
      return (await base.getStream()).filter(filterer);
    });
  }

  /**
   * Maps this dataset through a 1-to-1 transform.
   *
   * @param transform A function mapping a `DatasetElement` to a transformed
   *   `DatasetElement`.
   *
   * @returns A `Dataset` of transformed elements.
   */
  map(transform: (value: DatasetElement) => DatasetElement |
          Promise<DatasetElement>): Dataset {
    const base = this;
    return datasetFromStreamFn(async () => {
      return (await base.getStream()).map(transform);
    });
  }

  /**
   * Groups elements into batches and arranges their values in columnar form.
   *
   * It is assumed that each of the incoming DatasetElements has the same set of
   * keys.  For each key, the resulting BatchDataset provides a BatchElement
   * collecting all of the incoming values for that key.  Incoming strings are
   * grouped into a string[].  Incoming Tensors are grouped into a new Tensor
   * where the 0'th axis is the batch dimension.  These columnar representations
   * for each key can be zipped together to reconstruct the original
   * DatasetElements.
   *
   * @param batchSize The number of elements desired per batch.
   * @param smallLastBatch Whether to emit the final batch when it has fewer
   *   than batchSize elements. Default true.
   * @returns A `BatchDataset`, from which a stream of batches can be obtained.
   */
  batch(batchSize: number, smallLastBatch = true): BatchDataset {
    return new BatchDataset(this, batchSize, smallLastBatch);
  }

  /**
   * Concatenates this `Dataset` with another.
   *
   * @param dataset A `Dataset` to be concatenated onto this one.
   * @returns A `Dataset`.
   */
  concatenate(dataset: Dataset): Dataset {
    const base = this;
    return datasetFromStreamFn(async () => {
      return (await base.getStream()).concatenate(await dataset.getStream());
    });
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
  repeat(count?: number): Dataset {
    const base = this;
    return datasetFromStreamFn(async () => {
      const streamStream = streamFromFunction(() => base.getStream());
      return (await streamFromConcatenated(streamStream.take(count)));
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
  take(count: number): Dataset {
    const base = this;
    return datasetFromStreamFn(async () => {
      return (await base.getStream()).take(count);
    });
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
  skip(count: number): Dataset {
    const base = this;
    return datasetFromStreamFn(async () => {
      return (await base.getStream()).skip(count);
    });
  }

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
      Dataset {
    const base = this;
    const random = seedrandom(seed);
    return datasetFromStreamFn(async () => {
      let seed2 = random.int32();
      if (reshuffleEachIteration) {
        seed2 += random.int32();
      }
      return (await base.getStream()).shuffle(bufferSize, seed2.toString());
    });
  }

  /**
   *  Creates a `Dataset` that prefetches elements from this Dataset.
   *
   * @param bufferSize: An integer specifying the number of elements to be
   *   prefetched.
   * @returns A `Dataset`.
   */
  prefetch(bufferSize: number): Dataset {
    const base = this;
    return datasetFromStreamFn(async () => {
      return (await base.getStream()).prefetch(bufferSize);
    });
  }

  /* TODO(soergel): for parity with tf.data:
  Dataset.flat_map()
  Dataset.zip()
  Dataset.dense_to_sparse_batch()
  Dataset.group_by_window()
  Dataset.padded_batch()
  */
}

/**
 * Create a `Dataset` defined by a provided getStream() function.
 */
export function datasetFromStreamFn(
    getStreamFn: () => Promise<DataStream<DatasetElement>>): Dataset {
  return new class extends Dataset {
    /*
     * Provide a new stream of elements.  Note this will also start new streams
     * from any underlying `Dataset`s.
     */
    async getStream(): Promise<DataStream<DatasetElement>> {
      return getStreamFn();
    }
  }
  ();
}

/**
 * Create a `Dataset` from an array of elements.
 */
export function datasetFromElements(items: DatasetElement[]): Dataset {
  return datasetFromStreamFn(async () => {
    return Promise.resolve(streamFromItems(items));
  });
}

/**
 * Create a `Dataset` by concatenating underlying `Dataset`s.
 *
 * Note that if the underlying `Dataset`s return elements in a
 * nondeterministic order, then this concatenated `Dataset` will do the same.
 */
export function datasetFromConcatenated(datasets: Dataset[]) {
  return datasetFromStreamFn(async () => {
    const streamStream = await Promise.all(datasets.map((d) => d.getStream()));
    return streamFromConcatenated(streamFromItems(streamStream));
  });
}
