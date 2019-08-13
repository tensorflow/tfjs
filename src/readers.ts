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

import {TensorContainer} from '@tensorflow/tfjs-core';
import {Dataset, datasetFromIteratorFn} from './dataset';
import {CSVDataset} from './datasets/csv_dataset';
import {iteratorFromFunction} from './iterators/lazy_iterator';
import {MicrophoneIterator} from './iterators/microphone_iterator';
import {WebcamIterator} from './iterators/webcam_iterator';
import {URLDataSource} from './sources/url_data_source';
import {CSVConfig, MicrophoneConfig, WebcamConfig} from './types';

/**
 * Create a `CSVDataset` by reading and decoding CSV file(s) from provided URL
 * or local path if it's in Node environment.
 *
 * Note: If isLabel in columnConfigs is `true` for at least one column, the
 * element in returned `CSVDataset` will be an object of
 * `{xs:features, ys:labels}`: xs is a dict of features key/value pairs, ys
 * is a dict of labels key/value pairs. If no column is marked as label,
 * returns a dict of features only.
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
 *     .map(({xs, ys}) =>
 *       {
 *         // Convert xs(features) and ys(labels) from object form (keyed by
 *         // column name) to array form.
 *         return {xs:Object.values(xs), ys:Object.values(ys)};
 *       })
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
 * @param source URL or local path to get CSV file. If it's a local path, it
 * must have prefix `file://` and it only works in node environment.
 * @param csvConfig (Optional) A CSVConfig object that contains configurations
 *     of reading and decoding from CSV file(s).
 */
/**
 * @doc {
 *   heading: 'Data',
 *   subheading: 'Creation',
 *   namespace: 'data',
 *   configParamIndices: [1]
 *  }
 */
export function csv(
    source: RequestInfo, csvConfig: CSVConfig = {}): CSVDataset {
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
 * const ds = tf.data.func(func);
 * await ds.forEachAsync(e => console.log(e));
 * ```
 *
 * @param f A function that produces one data element on each call.
 */
export function func<T extends TensorContainer>(
    f: () => IteratorResult<T>| Promise<IteratorResult<T>>): Dataset<T> {
  const iter = iteratorFromFunction(f);
  return datasetFromIteratorFn(async () => iter);
}

/**
 * Create a `Dataset` that produces each element from provided JavaScript
 * generator, which is a function*
 * (https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Iterators_and_Generators#Generator_functions),
 * or a function that returns an
 * iterator
 * (https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Iterators_and_Generators#Generator_functions).
 *
 * The returned iterator should have `.next()` function that returns element in
 * format of `{value: TensorContainer, done:boolean}`.
 *
 * Example of creating a dataset from an iterator factory:
 * ```js
 * function makeIterator() {
 *   const numElements = 10;
 *   let index = 0;
 *
 *   const iterator = {
 *     next: () => {
 *       let result;
 *       if (index < numElements) {
 *         result = {value: index, done: false};
 *         index++;
 *         return result;
 *       }
 *       return {value: index, done: true};
 *     }
 *   };
 *   return iterator;
 * }
 * const ds = tf.data.generator(makeIterator);
 * ds.forEachAsync(e => console.log(e));
 * ```
 *
 * Example of creating a dataset from a generator:
 * ```js
 * function* dataGenerator() {
 *   const numElements = 10;
 *   let index = 0;
 *   while (index < numElements) {
 *     const x = index;
 *     index++;
 *     yield x;
 *   }
 * }
 *
 * const ds = tf.data.generator(dataGenerator);
 * ds.forEachAsync(e => console.log(e));
 * ```
 *
 * @param generator A Javascript generator function that returns a JavaScript
 *     iterator.
 */
/**
 * @doc {
 *   heading: 'Data',
 *   subheading: 'Creation',
 *   namespace: 'data',
 *   configParamIndices: [1]
 *  }
 */
export function generator<T extends TensorContainer>(
    generator: () => Iterator<T>| Promise<Iterator<T>>): Dataset<T> {
  return datasetFromIteratorFn(async () => {
    const gen = await generator();
    return iteratorFromFunction(() => gen.next());
  });
}

/**
 * Create an iterator that generate `Tensor`s from webcam video stream. This API
 * only works in Browser environment when the device has webcam.
 *
 * Note: this code snippet only works when the device has a webcam. It will
 * request permission to open the webcam when running.
 * ```js
 * const videoElement = document.createElement('video');
 * videoElement.width = 100;
 * videoElement.height = 100;
 * const cam = await tf.data.webcam(videoElement);
 * const img = await cam.capture();
 * img.print();
 * cam.stop();
 * ```
 *
 * @param webcamVideoElement A `HTMLVideoElement` used to play video from
 *     webcam. If this element is not provided, a hidden `HTMLVideoElement` will
 *     be created. In that case, `resizeWidth` and `resizeHeight` must be
 *     provided to set the generated tensor shape.
 * @param webcamConfig A `WebcamConfig` object that contains configurations of
 *     reading and manipulating data from webcam video stream.
 */
/**
 * @doc {
 *   heading: 'Data',
 *   subheading: 'Creation',
 *   namespace: 'data',
 *   ignoreCI: true
 *  }
 */
export async function webcam(
    webcamVideoElement?: HTMLVideoElement,
    webcamConfig?: WebcamConfig): Promise<WebcamIterator> {
  return WebcamIterator.create(webcamVideoElement, webcamConfig);
}

/**
 * Create an iterator that generate frequency-domain spectrogram `Tensor`s from
 * microphone audio stream with browser's native FFT. This API only works in
 * browser environment when the device has microphone.
 *
 * Note: this code snippet only works when the device has a microphone. It will
 * request permission to open the microphone when running.
 * ```js
 * const mic = await tf.data.microphone({
 *   fftSize: 1024,
 *   columnTruncateLength: 232,
 *   numFramesPerSpectrogram: 43,
 *   sampleRateHz:44100,
 *   includeSpectrogram: true,
 *   includeWaveform: true
 * });
 * const audioData = await mic.capture();
 * const spectrogramTensor = audioData.spectrogram;
 * spectrogramTensor.print();
 * const waveformTensor = audioData.waveform;
 * waveformTensor.print();
 * mic.stop();
 * ```
 *
 * @param microphoneConfig A `MicrophoneConfig` object that contains
 *     configurations of reading audio data from microphone.
 */
/**
 * @doc {
 *   heading: 'Data',
 *   subheading: 'Creation',
 *   namespace: 'data',
 *   ignoreCI: true
 *  }
 */
export async function microphone(microphoneConfig?: MicrophoneConfig):
    Promise<MicrophoneIterator> {
  return MicrophoneIterator.create(microphoneConfig);
}
