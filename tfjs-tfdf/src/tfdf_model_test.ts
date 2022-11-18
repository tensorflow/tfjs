/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';

import {loadTFDFModel, TFDFModel} from './tfdf_model';

describe('TFDFModel ', () => {
  let model: TFDFModel;
  beforeEach(async () => {
    model = await loadTFDFModel(
        'https://storage.googleapis.com/tfjs-testing/adult_gbt_no_regex/model.json');
  });

  it('predict', async () => {
    const inputs = {
      'age': tf.tensor1d([39, 40, 40, 35], 'int32'),
      'workclass': tf.tensor1d(
          ['State-gov', 'Private', 'Private', 'Federal-gov'], 'string'),
      'fnlwgt': tf.tensor1d([77516, 121772, 193524, 76845], 'int32'),
      'education':
          tf.tensor1d(['Bachelors', 'Assoc-voc', 'Doctorate', '9th'], 'string'),
      'education_num': tf.tensor1d([13, 11, 16, 5], 'int32'),
      'marital_status': tf.tensor1d(
          [
            'Never-married', 'Married-civ-spouse', 'Married-civ-spouse',
            'Married-civ-spouse'
          ],
          'string'),
      'occupation': tf.tensor1d(
          ['Adm-clerical', 'Craft-repair', 'Prof-specialty', 'Farming-fishing'],
          'string'),
      'relationship': tf.tensor1d(
          ['Not-in-family', 'Husband', 'Husband', 'Husband'], 'string'),
      'race': tf.tensor1d(
          ['White', 'Asian-Pac-Islander', 'White', 'Black'], 'string'),
      'sex': tf.tensor1d(['Male', 'Male', 'Male', 'Male'], 'string'),
      'capital_gain': tf.tensor1d([2174, 0, 0, 0], 'int32'),
      'capital_loss': tf.tensor1d([0, 0, 0, 0], 'int32'),
      'hours_per_week': tf.tensor1d([40, 40, 60, 40], 'int32'),
      'native_country': tf.tensor1d(
          ['United-States', '', 'United-States', 'United-States'], 'string')
    };

    const densePredictions = await model.executeAsync(inputs) as tf.Tensor;

    tf.test_util.expectArraysEqual(densePredictions.shape, [4, 1]);
    tf.test_util.expectArraysClose(await densePredictions.data(), [[
                                     0.13323983550071716,
                                     0.47678571939468384,
                                     0.818461537361145,
                                     0.4974619150161743,
                                   ]]);
  });
});
