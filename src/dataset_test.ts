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
import {TensorContainerObject} from '@tensorflow/tfjs-core/dist/types';

import {Dataset, datasetFromElements} from './dataset';
// import {CPU_ENVS, describeWithFlags} from '../../test_util';
import {iteratorFromItems, LazyIterator} from './streams/lazy_iterator';

class TestObjectStream extends LazyIterator<{}> {
  data = Array.from({length: 100}, (v, k) => k);
  currentIndex = 0;

  async next(): Promise<IteratorResult<{}>> {
    if (this.currentIndex >= 100) {
      return {value: null, done: true};
    }
    const elementNumber = this.data[this.currentIndex];
    const result = {
      'number': elementNumber,
      'numberArray': [elementNumber, elementNumber ** 2, elementNumber ** 3],
      'Tensor':
          tf.tensor1d([elementNumber, elementNumber ** 2, elementNumber ** 3]),
      'Tensor2': tf.tensor2d(
          [
            elementNumber, elementNumber ** 2, elementNumber ** 3,
            elementNumber ** 4
          ],
          [2, 2]),
      'string': `Item ${elementNumber}`
    };
    this.currentIndex++;
    return {value: result, done: false};
  }
}

export class TestDataset extends Dataset<TensorContainerObject> {
  iterator(): LazyIterator<{}> {
    return new TestObjectStream();
  }
}

tf.test_util.describeWithFlags('Dataset', tf.test_util.CPU_ENVS, () => {
  it('can be concatenated', done => {
    const a = datasetFromElements([{'item': 1}, {'item': 2}, {'item': 3}]);
    const b = datasetFromElements([{'item': 4}, {'item': 5}, {'item': 6}]);
    a.concatenate(b)
        .collectAll()
        .then(result => {
          expect(result).toEqual([
            {'item': 1}, {'item': 2}, {'item': 3}, {'item': 4}, {'item': 5},
            {'item': 6}
          ]);
        })
        .then(done)
        .catch(done.fail);
  });

  it('can be created by concatenating multiple underlying datasets via reduce',
     async done => {
       const a = datasetFromElements([{'item': 1}, {'item': 2}]);
       const b = datasetFromElements([{'item': 3}, {'item': 4}]);
       const c = datasetFromElements([{'item': 5}, {'item': 6}]);
       const concatenated = [a, b, c].reduce((a, b) => a.concatenate(b));
       concatenated.collectAll()
           .then(result => {
             expect(result).toEqual([
               {'item': 1}, {'item': 2}, {'item': 3}, {'item': 4}, {'item': 5},
               {'item': 6}
             ]);
           })
           .then(done)
           .catch(done.fail);
     });

  it('can be repeated a fixed number of times', done => {
    const a = datasetFromElements([{'item': 1}, {'item': 2}, {'item': 3}]);
    a.repeat(4)
        .collectAll()
        .then(result => {
          expect(result).toEqual([
            {'item': 1},
            {'item': 2},
            {'item': 3},
            {'item': 1},
            {'item': 2},
            {'item': 3},
            {'item': 1},
            {'item': 2},
            {'item': 3},
            {'item': 1},
            {'item': 2},
            {'item': 3},
          ]);
        })
        .then(done)
        .catch(done.fail);
  });

  it('can be repeated indefinitely', done => {
    const a = datasetFromElements([{'item': 1}, {'item': 2}, {'item': 3}]);
    a.repeat().take(234).collectAll().then(done).catch(done.fail);
    done();
  });

  it('can be repeated with state in a closure', done => {
    // This tests a tricky bug having to do with 'this' being set properly.
    // See https://github.com/Microsoft/TypeScript/wiki/%27this%27-in-TypeScript

    class CustomDataset extends Dataset<{}> {
      state = {val: 1};
      iterator() {
        const result = iteratorFromItems([
          {'item': this.state.val++}, {'item': this.state.val++},
          {'item': this.state.val++}
        ]);
        return result;
      }
    }
    const a = new CustomDataset();
    a.repeat().take(1234).collectAll().then(done).catch(done.fail);
  });

  it('can collect all items into memory', async done => {
    try {
      const ds = new TestDataset();
      const items = await ds.collectAll();
      expect(items.length).toEqual(100);
      // The test dataset has 100 elements, each containing 2 Tensors.
      expect(tf.memory().numTensors).toEqual(200);
      done();
    } catch (e) {
      done.fail(e);
    }
  });

  it('skip does not leak Tensors', async done => {
    try {
      const ds = new TestDataset();
      expect(tf.memory().numTensors).toEqual(0);
      const result = await ds.skip(15).collectAll();
      // The test dataset had 100 elements; we skipped 15; 85 remain.
      expect(result.length).toEqual(85);
      // Each element of the test dataset contains 2 Tensors;
      // 85 elements remain, so 2 * 85 = 170 Tensors remain.
      expect(tf.memory().numTensors).toEqual(170);
      done();
    } catch (e) {
      done.fail(e);
    }
  });

  it('filter does not leak Tensors', async done => {
    try {
      const ds = new TestDataset();
      expect(tf.memory().numTensors).toEqual(0);
      await ds.filter(x => ((x['number'] as number) % 2 === 0)).collectAll();
      // Each element of the test dataset contains 2 Tensors.
      // There were 100 elements, but we filtered out half of them.
      // Thus 50 * 2 = 100 Tensors remain.
      expect(tf.memory().numTensors).toEqual(100);
      done();
    } catch (e) {
      done.fail(e);
    }
  });

  it('map does not leak Tensors when none are returned', async done => {
    try {
      const ds = new TestDataset();
      expect(tf.memory().numTensors).toEqual(0);
      await ds.map(x => ({'constant': 1})).collectAll();
      // The map operation consumed all of the tensors and emitted none.
      expect(tf.memory().numTensors).toEqual(0);
      done();
    } catch (e) {
      done.fail(e);
    }
  });

  it('map does not lose or leak Tensors when some inputs are passed through',
     async done => {
       try {
         const ds = new TestDataset();
         expect(tf.memory().numTensors).toEqual(0);
         await ds.map(x => ({'Tensor2': x['Tensor2']})).collectAll();
         // Each element of the test dataset contains 2 Tensors.
         // Our map operation retained one of the Tensors and discarded the
         // other. Thus the mapped data contains 100 elements with 1 Tensor
         // each.
         expect(tf.memory().numTensors).toEqual(100);
         done();
       } catch (e) {
         done.fail(e);
       }
     });

  it('map does not leak Tensors when inputs are replaced', async done => {
    try {
      const ds = new TestDataset();
      expect(tf.memory().numTensors).toEqual(0);
      await ds.map(x => ({'a': tf.tensor1d([1, 2, 3])})).collectAll();
      // Each element of the test dataset contains 2 Tensors.
      // Our map operation discarded both Tensors and created one new one.
      // Thus the mapped data contains 100 elements with 1 Tensor each.
      expect(tf.memory().numTensors).toEqual(100);
      done();
    } catch (e) {
      done.fail(e);
    }
  });

  it('forEach does not leak Tensors', async done => {
    try {
      const ds = new TestDataset();
      let count = 0;
      await ds.forEach(element => {
        count++;
        return {};
      });
      // forEach traversed the entire dataset of 100 elements.
      expect(count).toEqual(100);
      // forEach consumed all of the input Tensors.
      expect(tf.memory().numTensors).toEqual(0);
      done();
    } catch (e) {
      done.fail(e);
    }
  });
});
