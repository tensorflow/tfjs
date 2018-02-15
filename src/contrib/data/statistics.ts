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

import {Scalar, Tensor} from '../../tensor';

import {Dataset} from './dataset';
import {ElementArray} from './types';

// TODO(soergel): Flesh out collected statistics.
// For numeric columns we should provide mean, stddev, histogram, etc.
// For string columns we should provide a vocabulary (at least, top-k), maybe a
// length histogram, etc.
// Collecting only numeric min and max is just the bare minimum for now.

export type NumericColumnStatistics = {
  min: number; max: number;
};

export type DatasetStatistics = {
  [key: string]: NumericColumnStatistics
};

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
  const minTensor: Tensor = Scalar.new(min);
  const rangeTensor: Tensor = Scalar.new(range);
  return (value: ElementArray): ElementArray => {
    if (typeof (value) === 'string') {
      throw new Error('Can\'t scale a string.');
    } else {
      if (value instanceof Tensor) {
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

export async function computeDatasetStatistics(
    dataset: Dataset, sampleSize?: number,
    shuffleWindowSize?: number): Promise<DatasetStatistics> {
  let stream = await dataset.getStream();
  // TODO(soergel): allow for deep shuffle where possible.
  if (shuffleWindowSize != null) {
    stream = stream.shuffle(shuffleWindowSize);
  }
  if (sampleSize != null) {
    stream = stream.take(sampleSize);
  }

  // TODO(soergel): prepare the column objects based on a schema.q
  const result: DatasetStatistics = {};

  await stream.forEach(e => {
    for (const key in e) {
      const value = e[key];
      if (typeof (value) === 'string') {
      } else {
        let recordMin: number;
        let recordMax: number;
        if (value instanceof Tensor) {
          recordMin = value.min().dataSync()[0];
          recordMax = value.max().dataSync()[0];
        } else if (value instanceof Array) {
          recordMin = value.reduce((a, b) => Math.min(a, b));
          recordMax = value.reduce((a, b) => Math.max(a, b));
        } else if (!isNaN(value) && isFinite(value)) {
          recordMin = value;
          recordMax = value;
        } else {
          // TODO(soergel): don't throw; instead record the stats as "unknown".
          throw new Error(`Cannot compute statistics: ${key} = ${value}`);
        }
        let columnStats: NumericColumnStatistics = result[key];
        if (columnStats == null) {
          columnStats = {
            min: Number.POSITIVE_INFINITY,
            max: Number.NEGATIVE_INFINITY
          };
          result[key] = columnStats;
        }
        columnStats.min = Math.min(columnStats.min, recordMin);
        columnStats.max = Math.max(columnStats.max, recordMax);
      }
    }
    // Returning undefined or null (i.e, type void) would indicate that the
    // stream is exhausted.  So, we have to return *something* in order for
    // resolveFully() to operate.
    return {};
  });
  return result;
}
