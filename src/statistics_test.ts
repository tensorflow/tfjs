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

import * as tf from '@tensorflow/tfjs-core';
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {Dataset} from '.';
import {TestDataset} from './dataset_test';
import {computeDatasetStatistics, scaleTo01} from './statistics';
import {TabularRecord} from './types';

describeWithFlags('makeDatasetStatistics', tf.test_util.ALL_ENVS, () => {
  it('computes numeric min and max over numbers, arrays, and Tensors', done => {
    const ds = new TestDataset().skip(55) as Dataset<TabularRecord>;
    computeDatasetStatistics(ds)
        .then(stats => {
          expect(stats['number'].min).toEqual(55);
          expect(stats['number'].max).toEqual(99);
          // The TestDataset includes cubes of the indices
          expect(stats['numberArray'].min).toEqual(55);
          expect(stats['numberArray'].max).toEqual(99 * 99 * 99);
          expect(stats['Tensor'].min).toEqual(55);
          expect(stats['Tensor'].max).toEqual(99 * 99 * 99);
        })
        .then(done)
        .catch(done.fail);
  });
});

describeWithFlags('scaleTo01', tf.test_util.ALL_ENVS, () => {
  it('scales numeric data to the [0, 1] interval', done => {
    const ds = new TestDataset().skip(55) as Dataset<TabularRecord>;
    const scaleFn = scaleTo01(55, 99 * 99 * 99);
    const scaledDataset = ds.map(x => ({'Tensor': scaleFn(x['Tensor'])}));

    computeDatasetStatistics(scaledDataset)
        .then(stats => {
          expect(stats['Tensor'].min).toBeCloseTo(0);
          expect(stats['Tensor'].max).toBeCloseTo(1);
        })
        .then(done)
        .catch(done.fail);
  });
});
