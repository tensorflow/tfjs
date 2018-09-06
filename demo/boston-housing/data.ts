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
import {CSVDataset, CsvHeaderConfig} from '../../src/datasets/csv_dataset';
import {URLDataSource} from '../../src/sources/url_data_source';
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
    await result.setData();
    return result;
  }

  /**
   * Downloads and returns the csv in array of numbers.
   */
  async loadCsv(filename: string) {
    const url = `${BASE_URL}${filename}`;

    console.log(`  * Downloading data from: ${url}`);

    const source = new URLDataSource(url);

    const dataset =
        await CSVDataset.create(source, CsvHeaderConfig.READ_FIRST_LINE);

    // Sets number of features so it can be used in the model.
    if (filename === TRAIN_FEATURES_FILENAME) {
      this.numFeatures = dataset.csvColumnNames.length;
    }

    // Reduces the object-type data to an array of numbers.
    return dataset.map((row: {[key: string]: string}) => {
      return Object.keys(row).sort().map(key => Number(row[key]));
    });
  }

  /**
   * Downloads, converts and shuffles the data.
   */
  private async setData() {
    const trainFeaturesDataset = await this.loadCsv(TRAIN_FEATURES_FILENAME);
    const trainTargetDataset = await this.loadCsv(TRAIN_TARGET_FILENAME);
    const testFeaturesDataset = await this.loadCsv(TEST_FEATURES_FILENAME);
    const testTargetDataset = await this.loadCsv(TEST_TARGET_FILENAME);

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
