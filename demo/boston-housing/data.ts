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
 * =============================================================================
 */

import {Dataset, zip} from '../../src/dataset';
import * as tfd from '../../src/readers';
import {DataElement} from '../../src/types';

// Boston Housing data constants:
const BASE_URL =
  'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/';

const TRAIN_FEATURES_FILENAME = 'train-data.csv';
const TRAIN_TARGET_FILENAME = 'train-target.csv';
const TEST_FEATURES_FILENAME = 'test-data.csv';
const TEST_TARGET_FILENAME = 'test-target.csv';

/** Helper class to handle loading training and test data. */
export class BostonHousingDataset {
  trainDataset: Dataset<DataElement> = null;
  testDataset: Dataset<DataElement> = null;
  numFeatures: number = null;

  private constructor() {}

  static async create() {
    const result = new BostonHousingDataset();
    await result.loadData();
    return result;
  }

  /**
   * Downloads, converts and shuffles the data.
   */
  private async loadData() {
    const fileUrls = [
      `${BASE_URL}${TRAIN_FEATURES_FILENAME}`,
      `${BASE_URL}${TRAIN_TARGET_FILENAME}`,
      `${BASE_URL}${TEST_FEATURES_FILENAME}`,
      `${BASE_URL}${TEST_TARGET_FILENAME}`
    ];
    console.log('* Downloading data *');
    const csvDatasets = fileUrls.map(url => tfd.csv(url, /* header */ true));

    // Sets number of features so it can be used in the model.
    this.numFeatures = (await csvDatasets[0]).csvColumnNames.length;

    // Reduces the object-type data to an array of numbers.
    const convertedDatasets = csvDatasets.map(
      async (dataset) =>
        (await dataset).map((row: {[key: string]: string}) => {
          return Object.keys(row).sort().map(key => Number(row[key]));
        }));

    const trainFeaturesDataset = await convertedDatasets[0];
    const trainTargetDataset = await convertedDatasets[1];
    const testFeaturesDataset = await convertedDatasets[2];
    const testTargetDataset = await convertedDatasets[3];

    this.trainDataset = await zip({
      features: trainFeaturesDataset,
      target: trainTargetDataset
    }).shuffle(1000);
    this.testDataset = await zip({
      features: testFeaturesDataset,
      target: testTargetDataset
    }).shuffle(1000);
  }
}
