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

import {TestDataset} from './dataset_test';

describeWithFlags('Dataset.batch()', tf.test_util.CPU_ENVS, () => {
  it('batches entries into column-oriented DatasetBatches', done => {
    const ds = new TestDataset();
    const bds = ds.batch(8);
    const batchIteratorPromise = bds.iterator();
    batchIteratorPromise
        .then(batchIterator => batchIterator.collect().then(result => {
          expect(result.length).toEqual(13);
          result.slice(0, 12).forEach(batch => {
            expect((batch['number'] as tf.Tensor).shape).toEqual([8]);
            expect((batch['numberArray'] as tf.Tensor).shape).toEqual([8, 3]);
            expect((batch['Tensor'] as tf.Tensor).shape).toEqual([8, 3]);
            expect((batch['string'] as string[]).length).toEqual(8);
          });
          return result;
        }))
        .then((result) => {
          result.forEach(tf.dispose);
        })
        .then(() => expect(tf.ENV.engine.memory().numTensors).toBe(0))
        .then(done)
        .catch(done.fail);
  });
  it('creates a small last batch', done => {
    const ds = new TestDataset();
    const bds = ds.batch(8);
    const batchIteratorPromise = bds.iterator();
    batchIteratorPromise
        .then(batchIterator => batchIterator.collect().then(result => {
          const lastBatch = result[12];
          expect((lastBatch['number'] as tf.Tensor).shape).toEqual([4]);
          expect((lastBatch['numberArray'] as tf.Tensor).shape).toEqual([4, 3]);
          expect((lastBatch['Tensor'] as tf.Tensor).shape).toEqual([4, 3]);
          expect((lastBatch['string'] as string[]).length).toEqual(4);

          tf.test_util.expectArraysClose(
              lastBatch['number'] as tf.Tensor, tf.tensor1d([96, 97, 98, 99]));
          tf.test_util.expectArraysClose(
              lastBatch['numberArray'] as tf.Tensor,
              tf.tensor2d(
                  [
                    [96, 96 ** 2, 96 ** 3], [97, 97 ** 2, 97 ** 3],
                    [98, 98 ** 2, 98 ** 3], [99, 99 ** 2, 99 ** 3]
                  ],
                  [4, 3]));
          tf.test_util.expectArraysClose(
              lastBatch['Tensor'] as tf.Tensor,
              tf.tensor2d(
                  [
                    [96, 96 ** 2, 96 ** 3], [97, 97 ** 2, 97 ** 3],
                    [98, 98 ** 2, 98 ** 3], [99, 99 ** 2, 99 ** 3]
                  ],
                  [4, 3]));
          expect(lastBatch['string'] as string[]).toEqual([
            'Item 96', 'Item 97', 'Item 98', 'Item 99'
          ]);

          expect(lastBatch['string'] as string[]).toEqual([
            'Item 96', 'Item 97', 'Item 98', 'Item 99'
          ]);
          return result;
        }))
        .then((result) => {
          result.forEach(tf.dispose);
        })
        // these three tensors are just the expected results above
        .then(() => expect(tf.ENV.engine.memory().numTensors).toBe(3))
        .then(done)
        .catch(done.fail);
  });
});
