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

import {Dataset} from './dataset';

// TODO(kangyizhang): eliminate the need for ElementArray and TabularRecord, by
// computing stats on nested structures via deepMap/deepZip.

/**
 * The value associated with a given key for a single element.
 *
 * Such a value may not have a batch dimension.  A value may be a scalar or an
 * n-dimensional array.
 */
export type ElementArray = number|number[]|tf.Tensor|string;

/**
 * A map from string keys (aka column names) to values for a single element.
 */
export type TabularRecord = {
  [key: string]: ElementArray
};

// TODO(kangyizhang): Flesh out collected statistics.
// For numeric columns we should provide mean, stddev, histogram, etc.
// For string columns we should provide a vocabulary (at least, top-k), maybe a
// length histogram, etc.
// Collecting only numeric min and max is just the bare minimum for now.

/** An interface representing numeric statistics of a column. */
export interface NumericColumnStatistics {
  min: number;
  max: number;
  mean: number;
  variance: number;
  stddev: number;
  length: number;
}

/**
 * An interface representing column level NumericColumnStatistics for a
 * Dataset.
 */
export interface DatasetStatistics {
  [key: string]: NumericColumnStatistics;
}

/**
 * Provides a function that scales numeric values into the [0, 1] interval.
 *
 * @param min the lower bound of the inputs, which should be mapped to 0.
 * @param max the upper bound of the inputs, which should be mapped to 1,
 * @return A function that maps an input ElementArray to a scaled ElementArray.
 */
export function scaleTo01(min: number, max: number): (value: ElementArray) =>
    ElementArray {
  const range = max - min;
  const minTensor: tf.Tensor = tf.scalar(min);
  const rangeTensor: tf.Tensor = tf.scalar(range);
  return (value: ElementArray): ElementArray => {
    if (typeof (value) === 'string') {
      throw new Error('Can\'t scale a string.');
    } else {
      if (value instanceof tf.Tensor) {
        const result = value.sub(minTensor).div(rangeTensor);
        return result;
      } else if (value instanceof Array) {
        return value.map(v => (v - min) / range);
      } else {
        return (value - min) / range;
      }
    }
  };
}

/**
 * Provides a function that calculates column level statistics, i.e. min, max,
 * variance, stddev.
 *
 * @param dataset The Dataset object whose statistics will be calculated.
 * @param sampleSize (Optional) If set, statistics will only be calculated
 *     against a subset of the whole data.
 * @param shuffleWindowSize (Optional) If set, shuffle provided dataset before
 *     calculating statistics.
 * @return A DatasetStatistics object that contains NumericColumnStatistics of
 *     each column.
 */
export async function computeDatasetStatistics(
    dataset: Dataset<TabularRecord>, sampleSize?: number,
    shuffleWindowSize?: number): Promise<DatasetStatistics> {
  let sampleDataset = dataset;
  // TODO(soergel): allow for deep shuffle where possible.
  if (shuffleWindowSize != null) {
    sampleDataset = sampleDataset.shuffle(shuffleWindowSize);
  }
  if (sampleSize != null) {
    sampleDataset = sampleDataset.take(sampleSize);
  }

  // TODO(soergel): prepare the column objects based on a schema.
  const result: DatasetStatistics = {};

  await sampleDataset.forEachAsync(e => {
    for (const key of Object.keys(e)) {
      const value = e[key];
      if (typeof (value) === 'string') {
        // No statistics for string element.
      } else {
        let previousMean = 0;
        let previousLength = 0;
        let previousVariance = 0;
        let columnStats: NumericColumnStatistics = result[key];
        if (columnStats == null) {
          columnStats = {
            min: Number.POSITIVE_INFINITY,
            max: Number.NEGATIVE_INFINITY,
            mean: 0,
            variance: 0,
            stddev: 0,
            length: 0
          };
          result[key] = columnStats;
        } else {
          previousMean = columnStats.mean;
          previousLength = columnStats.length;
          previousVariance = columnStats.variance;
        }
        let recordMin: number;
        let recordMax: number;

        // Calculate accumulated mean and variance following tf.Transform
        // implementation
        let valueLength = 0;
        let valueMean = 0;
        let valueVariance = 0;
        let combinedLength = 0;
        let combinedMean = 0;
        let combinedVariance = 0;

        if (value instanceof tf.Tensor) {
          recordMin = value.min().dataSync()[0];
          recordMax = value.max().dataSync()[0];
          const valueMoment = tf.moments(value);
          valueMean = valueMoment.mean.dataSync()[0];
          valueVariance = valueMoment.variance.dataSync()[0];
          valueLength = value.size;

        } else if (value instanceof Array) {
          recordMin = value.reduce((a, b) => Math.min(a, b));
          recordMax = value.reduce((a, b) => Math.max(a, b));
          const valueMoment = tf.moments(value);
          valueMean = valueMoment.mean.dataSync()[0];
          valueVariance = valueMoment.variance.dataSync()[0];
          valueLength = value.length;

        } else if (!isNaN(value) && isFinite(value)) {
          recordMin = value;
          recordMax = value;
          valueMean = value;
          valueVariance = 0;
          valueLength = 1;

        } else {
          columnStats = null;
          continue;
        }
        combinedLength = previousLength + valueLength;
        combinedMean = previousMean +
            (valueLength / combinedLength) * (valueMean - previousMean);
        combinedVariance = previousVariance +
            (valueLength / combinedLength) *
                (valueVariance +
                 ((valueMean - combinedMean) * (valueMean - previousMean)) -
                 previousVariance);

        columnStats.min = Math.min(columnStats.min, recordMin);
        columnStats.max = Math.max(columnStats.max, recordMax);
        columnStats.length = combinedLength;
        columnStats.mean = combinedMean;
        columnStats.variance = combinedVariance;
        columnStats.stddev = Math.sqrt(combinedVariance);
      }
    }
  });
  // Variance and stddev should be NaN for the case of a single element.
  for (const key in result) {
    const stat: NumericColumnStatistics = result[key];
    if (stat.length === 1) {
      stat.variance = NaN;
      stat.stddev = NaN;
    }
  }
  return result;
}
