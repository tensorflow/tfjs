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

import * as tfd from '@tensorflow/tfjs-data';

// Boston Housing data constants:
const BASE_URL =
    'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/';

const TRAIN_FILENAME = 'boston-housing-train.csv';
const TEST_FILENAME = 'boston-housing-test.csv';

/** Helper class to handle loading training and test data. */
export class BostonHousingDataset {
  trainDataset: tfd.Dataset<tfd.DataElement> = null;
  testDataset: tfd.Dataset<tfd.DataElement> = null;
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
    console.log('* Downloading data *');

    const trainData = await this.prepareDataset(`${BASE_URL}${TRAIN_FILENAME}`);
    // Sets number of features so it can be used in the model. Need to exclude
    // the column of label.
    this.trainDataset = trainData.dataset;
    this.numFeatures = trainData.numFeatures;
    this.testDataset =
        (await this.prepareDataset(`${BASE_URL}${TEST_FILENAME}`)).dataset;
  }

  /**
   * Prepare dataset from provided url.
   */
  private async prepareDataset(url: string) {
    // We want to predict the column "medv", which represents a median value of
    // a home (in $1000s), so we mark it as a label.
    const csvDataset = tfd.csv(url, {columnConfigs: {medv: {isLabel: true}}});

    // Reduces the object-type data to an array of numbers.
    const convertedDataset =
        csvDataset.map((row: Array<{[key: string]: number}>) => {
          const [rawFeatures, rawLabel] = row;
          const convertedFeatures = Object.values(rawFeatures);
          const convertedLabel = [rawLabel['medv']];
          return [convertedFeatures, convertedLabel];
        });
    return {
      dataset: convertedDataset.shuffle(100),
      numFeatures: (await csvDataset.columnNames()).length - 1
    };
  }
}
