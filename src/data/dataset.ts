/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
 * =============================================================================
 */

import {NDArray} from '../math/ndarray';
import * as util from '../util';

const STATS_SAMPLE_PERCENTAGE = 0.1;

export interface DataStats {
  exampleCount: number;
  inputMin: number;
  inputMax: number;
  shape: number[];
}

interface NormalizationInfo {
  isNormalized: boolean;
  // Bounds of the normalization if normalized.
  lowerBound?: number;
  upperBound?: number;
  // Minimum and maximum values for each dimension of the original data. These
  // are the same size as an input example. These are computed lazily, only if
  // normalization is requested. If the data is un-normalized, these are kept
  // around so they don't have to be recomputed.
  minValues: Float32Array;
  maxValues: Float32Array;
}

export abstract class InMemoryDataset {
  protected dataset: NDArray[][]|null;

  // Contains information necessary for reconstruction of the original data
  // after normalization.
  private normalizationInfo: {[dataIndex: number]: NormalizationInfo};

  constructor(protected dataShapes: number[][]) {
    this.normalizationInfo = {};
  }

  getDataShape(dataIndex: number): number[] {
    return this.dataShapes[dataIndex];
  }

  abstract fetchData(): Promise<void>;

  getData(): NDArray[][]|null {
    return this.dataset;
  }

  getStats(): DataStats[] {
    if (this.dataset == null) {
      throw new Error('Data is null.');
    }

    return this.dataset.map(d => this.getStatsForData(d));
  }

  // Computes stats across a sampled portion of the data.
  private getStatsForData(data: NDArray[]): DataStats {
    let inputMin = Number.POSITIVE_INFINITY;
    let inputMax = Number.NEGATIVE_INFINITY;

    let exampleIndices = data.map((example, i) => i);
    util.shuffle(exampleIndices);
    exampleIndices =
        exampleIndices.slice(exampleIndices.length * STATS_SAMPLE_PERCENTAGE);

    for (let i = 0; i < exampleIndices.length; i++) {
      const inputValues = data[exampleIndices[i]].dataSync();
      for (let j = 0; j < inputValues.length; j++) {
        inputMin = Math.min(inputMin, inputValues[j]);
        inputMax = Math.max(inputMax, inputValues[j]);
      }
    }

    return {
      inputMin,
      inputMax,
      exampleCount: data.length,
      shape: data[0].shape,
    };
  }

  /**
   * @param examples NDArrays to be normalized.
   * @param curLowerBounds An array containing the minimum value for each
   * dimension or a fixed minimum value.
   * @param curUpperBounds An array containing the maximum value for each
   * dimension or a fixed maximum value.
   * @param newLowerBounds An array containing new minimum values for each
   * dimension, or a fixed minumum value to normalize the data to.
   * @param newUpperBounds An array containing new maximum values for each
   * dimension, or a fixed maximum value to normalize the data to.
   */
  private normalizeExamplesToRange(
      examples: NDArray[], curLowerBounds: Float32Array|number,
      curUpperBounds: Float32Array|number, newLowerBounds: Float32Array|number,
      newUpperBounds: Float32Array|number): NDArray[] {
    const curBoundsIsPerDimension =
        (curUpperBounds instanceof Float32Array &&
         curLowerBounds instanceof Float32Array);
    const newBoundsIsPerDimension =
        (newLowerBounds instanceof Float32Array &&
         newUpperBounds instanceof Float32Array);

    const inputSize = util.sizeFromShape(examples[0].shape);
    const newExamples: NDArray[] = [];

    examples.forEach(example => {
      const inputValues = example.dataSync();
      const normalizedValues = new Float32Array(inputSize);
      for (let j = 0; j < inputSize; j++) {
        const curLowerBound = curBoundsIsPerDimension ?
            (curLowerBounds as Float32Array)[j] :
            curLowerBounds as number;
        const curUpperBound = curBoundsIsPerDimension ?
            (curUpperBounds as Float32Array)[j] :
            curUpperBounds as number;
        const curRange = curUpperBound - curLowerBound;

        const newLowerBound = newBoundsIsPerDimension ?
            (newLowerBounds as Float32Array)[j] :
            newLowerBounds as number;
        const newUpperBound = newBoundsIsPerDimension ?
            (newUpperBounds as Float32Array)[j] :
            newUpperBounds as number;
        const newRange = newUpperBound - newLowerBound;

        if (curRange === 0) {
          normalizedValues[j] = newLowerBound;
        } else {
          normalizedValues[j] = newLowerBound +
              newRange * (inputValues[j] - curLowerBound) / curRange;
        }
      }
      newExamples.push(
          NDArray.make(example.shape, {values: normalizedValues}, 'float32'));
    });
    return newExamples;
  }

  private computeBounds(dataIndex: number) {
    if (this.dataset == null) {
      throw new Error('Data is null.');
    }

    const size = util.sizeFromShape(this.dataset[dataIndex][0].shape);

    // Compute min and max values for every dimension.
    this.normalizationInfo[dataIndex] = {
      isNormalized: false,
      minValues: new Float32Array(size),
      maxValues: new Float32Array(size)
    };

    for (let i = 0; i < size; i++) {
      this.normalizationInfo[dataIndex].minValues[i] = Number.POSITIVE_INFINITY;
      this.normalizationInfo[dataIndex].maxValues[i] = Number.NEGATIVE_INFINITY;
    }

    this.dataset[dataIndex].forEach(example => {
      const inputValues = example.dataSync();
      for (let k = 0; k < size; k++) {
        this.normalizationInfo[dataIndex].minValues[k] = Math.min(
            this.normalizationInfo[dataIndex].minValues[k], inputValues[k]);
        this.normalizationInfo[dataIndex].maxValues[k] = Math.max(
            this.normalizationInfo[dataIndex].maxValues[k], inputValues[k]);
      }
    });
  }

  normalizeWithinBounds(
      dataIndex: number, lowerBound: number, upperBound: number) {
    if (this.dataset == null) {
      throw new Error('Data is null.');
    }
    if (dataIndex >= this.dataset.length) {
      throw new Error('dataIndex out of bounds.');
    }

    if (this.normalizationInfo[dataIndex] == null) {
      this.computeBounds(dataIndex);
    }

    // curLower/UpperBounds of the current data set can either be fixed numbers
    // if the data has already been normalized, or curLower/Upper for each
    // dimension if it hasn't been normalized yet.
    let curLowerBounds: Float32Array|number;
    let curUpperBounds: Float32Array|number;

    if (this.normalizationInfo[dataIndex].isNormalized) {
      curLowerBounds = this.normalizationInfo[dataIndex].lowerBound;
      curUpperBounds = this.normalizationInfo[dataIndex].upperBound;
    } else {
      curLowerBounds = this.normalizationInfo[dataIndex].minValues;
      curUpperBounds = this.normalizationInfo[dataIndex].maxValues;
    }

    this.dataset[dataIndex] = this.normalizeExamplesToRange(
        this.dataset[dataIndex], curLowerBounds, curUpperBounds, lowerBound,
        upperBound);
    this.normalizationInfo[dataIndex].isNormalized = true;
    this.normalizationInfo[dataIndex].lowerBound = lowerBound;
    this.normalizationInfo[dataIndex].upperBound = upperBound;
  }

  private isNormalized(dataIndex: number): boolean {
    return this.normalizationInfo != null &&
        this.normalizationInfo[dataIndex].isNormalized;
  }

  removeNormalization(dataIndex: number) {
    if (this.dataset == null) {
      throw new Error('Training or test data is null.');
    }

    if (!this.isNormalized(dataIndex)) {
      return;
    }

    this.dataset[dataIndex] = this.normalizeExamplesToRange(
        this.dataset[dataIndex], this.normalizationInfo[dataIndex].lowerBound,
        this.normalizationInfo[dataIndex].upperBound,
        this.normalizationInfo[dataIndex].minValues,
        this.normalizationInfo[dataIndex].maxValues);
    this.normalizationInfo[dataIndex].isNormalized = false;
  }

  unnormalizeExamples(examples: NDArray[], dataIndex: number): NDArray[] {
    if (!this.isNormalized(dataIndex)) {
      return examples;
    }

    return this.normalizeExamplesToRange(
        examples, this.normalizationInfo[dataIndex].lowerBound,
        this.normalizationInfo[dataIndex].upperBound,
        this.normalizationInfo[dataIndex].minValues,
        this.normalizationInfo[dataIndex].maxValues);
  }

  dispose() {
    if (this.dataset == null) {
      return;
    }

    for (let i = 0; i < this.dataset.length; i++) {
      for (let j = 0; j < this.dataset[i].length; j++) {
        this.dataset[i][j].dispose();
      }
    }
    this.dataset = [];
  }
}
