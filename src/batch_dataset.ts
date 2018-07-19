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

// tslint:disable:max-line-length
import {Dataset} from './dataset';
import {LazyIterator} from './iterators/lazy_iterator';
import {BatchArray, DataElement, DatasetBatch, ElementArray, TabularRecord} from './types';
// tslint:enable:max-line-length

// TODO(soergel): refactor to remove BatchDataset class, but retain columnar
// batching functionality.

/**
 * Represents a potentially large set of elements, grouped into batches.
 *
 * There are currently no batch-oriented data transformations.  Any desired
 * transformations should be applied to a `Dataset` so that they are
 * computed one example at a time.  The transformed data can then be batched
 * as the final step via `Dataset.batch()`.
 *
 * @param base: An underlying row-oriented `Dataset` to group into batches.
 * @param batchSize: The desired number of examples per batch.
 * @param smallLastBatch: Whether to emit a final batch with fewer than
 *   batchSize elements.  (Default true).
 */
export class BatchDataset {
  constructor(
      protected base: Dataset<DataElement>, protected batchSize: number,
      protected smallLastBatch = true) {}

  /*
   * Provide a new stream of batches.  Note this will also start new streams
   * from any underlying `Dataset`s or 'BatchDataset's.
   */
  async iterator(): Promise<LazyIterator<DatasetBatch>> {
    const batchesAsArrays =
        (await this.base.iterator()).batch(this.batchSize, this.smallLastBatch);
    return batchesAsArrays.map(makeDatasetBatch);
  }
}

/**
 * Constructs a DatasetBatch from a list of TabularRecords.
 */
function makeDatasetBatch(elements: TabularRecord[]): DatasetBatch {
  const rotated: {[key: string]: (ElementArray[]|string[])} = {};

  // Assume that the first element is representative.
  // We do end up enforcing Tensor shape consistency below, but not
  // cleanly.
  // TODO(soergel) validate against a schema, allow missing keys, etc.
  // etc.
  const firstElement: TabularRecord = elements[0];
  const keys = Object.keys(firstElement);
  keys.forEach(key => {
    rotated[key] = [];
  });

  for (const e of elements) {
    keys.forEach(key => {
      const value = e[key];
      (rotated[key] as ElementArray[]).push(value);
    });
  }

  const result: {[key: string]: (BatchArray|string[])} = {};
  keys.forEach(key => {
    // this sanity check should always pass
    if (rotated[key].length !== elements.length) {
      throw new Error(
          `Batching failed to get a '${key}' value for each element.`);
    }
    if (typeof rotated[key][0] === 'string') {
      result[key] = rotated[key] as string[];
    } else {
      result[key] =
          batchConcat(rotated[key] as Array<number|number[]|tf.Tensor>);
    }
  });
  elements.forEach(tf.dispose);

  return result;
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
  const result = tf.Tensor.make(batchShape, {values: resultVals});
  return result;
}

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
