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

import {Dataset, datasetFromIteratorFn} from './dataset';
import {CSVDataset} from './datasets/csv_dataset';
import {iteratorFromFunction} from './iterators/lazy_iterator';
import {URLDataSource} from './sources/url_data_source';
import {CSVConfig, DataElement} from './types';

/**
 * Create a `CSVDataset` by reading and decoding CSV file(s) from provided URLs.
 *
 * ```js
 * const csvUrl =
 * 'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv';
 *
 * async function run() {
 *   // We want to predict the column "medv", which represents a median value of
 *   // a home (in $1000s), so we mark it as a label.
 *   const csvDataset = tf.data.csv(
 *     csvUrl, {
 *       columnConfigs: {
 *         medv: {
 *           isLabel: true
 *         }
 *       }
 *     });
 *
 *   // Number of features is the number of column names minus one for the label
 *   // column.
 *   const numOfFeatures = (await csvDataset.columnNames()).length - 1;
 *
 *   // Prepare the Dataset for training.
 *   const flattenedDataset =
 *     csvDataset
 *     .map(([rawFeatures, rawLabel]) =>
 *       // Convert rows from object form (keyed by column name) to array form.
 *       [Object.values(rawFeatures), Object.values(rawLabel)])
 *     .batch(10);
 *
 *   // Define the model.
 *   const model = tf.sequential();
 *   model.add(tf.layers.dense({
 *     inputShape: [numOfFeatures],
 *     units: 1
 *   }));
 *   model.compile({
 *     optimizer: tf.train.sgd(0.000001),
 *     loss: 'meanSquaredError'
 *   });
 *
 *   // Fit the model using the prepared Dataset
 *   return model.fitDataset(flattenedDataset, {
 *     epochs: 10,
 *     callbacks: {
 *       onEpochEnd: async (epoch, logs) => {
 *         console.log(epoch + ':' + logs.loss);
 *       }
 *     }
 *   });
 * }
 *
 * await run();
 * ```
 *
 * @param source URL to fetch CSV file.
 * @param csvConfig (Optional) A CSVConfig object that contains configurations
 *     of reading and decoding from CSV file(s).
 */
/** @doc {
 *   heading: 'Data',
 *   subheading: 'Creation',
 *   namespace: 'data',
 *   configParamIndices: [1]
 *  }
 */
export function csv(source: string, csvConfig: CSVConfig = {}): CSVDataset {
  return new CSVDataset(new URLDataSource(source), csvConfig);
}

/**
 * Create a `Dataset` that produces each element by calling a provided function.
 *
 * Note that repeated iterations over this `Dataset` may produce different
 * results, because the function will be called anew for each element of each
 * iteration.
 *
 * Also, beware that the sequence of calls to this function may be out of order
 * in time with respect to the logical order of the Dataset. This is due to the
 * asynchronous lazy nature of stream processing, and depends on downstream
 * transformations (e.g. .shuffle()). If the provided function is pure, this is
 * no problem, but if it is a closure over a mutable state (e.g., a traversal
 * pointer), then the order of the produced elements may be scrambled.
 *
 * ```js
 * let i = -1;
 * const func = () =>
 *    ++i < 5 ? {value: i, done: false} : {value: null, done: true};
 * const ds = tf.data.generator(func);
 * await ds.forEach(e => console.log(e));
 * ```
 *
 * @param f A function that produces one data element on each call.
 */
/** @doc {
 *   heading: 'Data',
 *   subheading: 'Creation',
 *   namespace: 'data',
 *   configParamIndices: [1]
 *  }
 */
export function generator<T extends DataElement>(
  f: () =>IteratorResult<T>| Promise<IteratorResult<T>>): Dataset<T> {
  const iter = iteratorFromFunction(f);
  return datasetFromIteratorFn(async () => iter);
}
