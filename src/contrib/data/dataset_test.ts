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

import {Tensor1D} from '../../tensor';

import {Dataset, datasetFromConcatenated, datasetFromElements} from './dataset';
import {DataStream, streamFromItems} from './streams/data_stream';
import {DatasetElement} from './types';

class TestDatasetElementStream extends DataStream<DatasetElement> {
  data = Array.from({length: 100}, (v, k) => k);
  currentIndex = 0;

  async next(): Promise<DatasetElement> {
    if (this.currentIndex >= 100) {
      return undefined;
    }
    const elementNumber = this.data[this.currentIndex];
    const result = {
      'number': elementNumber,
      'numberArray': [elementNumber, elementNumber ** 2, elementNumber ** 3],
      'Tensor':
          Tensor1D.new([elementNumber, elementNumber ** 2, elementNumber ** 3]),
      'string': `Item ${elementNumber}`
    };

    this.currentIndex++;
    return result;
  }
}

export class TestDataset extends Dataset {
  async getStream(): Promise<DataStream<DatasetElement>> {
    return new TestDatasetElementStream();
  }
}

describe('Dataset', () => {
  it('can be created by concatenating underlying datasets', done => {
    const a = datasetFromElements([{'item': 1}, {'item': 2}]);
    const b = datasetFromElements([{'item': 3}, {'item': 4}]);
    const c = datasetFromElements([{'item': 5}, {'item': 6}]);
    const readStreamPromise = datasetFromConcatenated([a, b, c]).getStream();
    readStreamPromise
        .then(readStream => readStream.collectRemaining().then(result => {
          expect(result).toEqual([
            {'item': 1}, {'item': 2}, {'item': 3}, {'item': 4}, {'item': 5},
            {'item': 6}
          ]);
        }))
        .then(done)
        .catch(done.fail);
  });

  it('can be concatenated', done => {
    const a = datasetFromElements([{'item': 1}, {'item': 2}, {'item': 3}]);
    const b = datasetFromElements([{'item': 4}, {'item': 5}, {'item': 6}]);
    const readStreamPromise = a.concatenate(b).getStream();
    readStreamPromise
        .then(readStream => readStream.collectRemaining().then(result => {
          expect(result).toEqual([
            {'item': 1}, {'item': 2}, {'item': 3}, {'item': 4}, {'item': 5},
            {'item': 6}
          ]);
        }))
        .then(done)
        .catch(done.fail);
  });

  it('can be repeated a fixed number of times', done => {
    const a = datasetFromElements([{'item': 1}, {'item': 2}, {'item': 3}]);
    const readStreamPromise = a.repeat(4).getStream();
    readStreamPromise
        .then(readStream => readStream.collectRemaining().then(result => {
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
        }))
        .then(done)
        .catch(done.fail);
  });

  it('can be repeated indefinitely', done => {
    const a = datasetFromElements([{'item': 1}, {'item': 2}, {'item': 3}]);
    const readStreamPromise = a.repeat().getStream();
    readStreamPromise
        .then(readStream => readStream.take(1234).collectRemaining())
        .then(done)
        .catch(done.fail);
    done();
  });

  it('can be repeated with state in a closure', done => {
    // This tests a tricky bug having to do with 'this' being set properly.
    // See https://github.com/Microsoft/TypeScript/wiki/%27this%27-in-TypeScript

    class CustomDataset extends Dataset {
      state = {val: 1};
      async getStream() {
        const result = streamFromItems([
          {'item': this.state.val++}, {'item': this.state.val++},
          {'item': this.state.val++}
        ]);
        return result;
      }
    }
    const a = new CustomDataset();
    const readStreamPromise = a.repeat().getStream();
    readStreamPromise
        .then(readStream => readStream.take(1234).collectRemaining())
        .then(done)
        .catch(done.fail);
    done();
  });
});
