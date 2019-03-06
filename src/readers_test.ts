/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import * as tfd from './readers';

describeWithFlags('readers', tf.test_util.ALL_ENVS, () => {
  it('generate dataset from function', async () => {
    let i = -1;
    const f = () =>
        ++i < 5 ? {value: i, done: false} : {value: null, done: true};
    const ds = tfd.func(f);
    const result = await ds.toArrayForTest();
    expect(result).toEqual([0, 1, 2, 3, 4]);
  });

  it('generate dataset from JavaScript generator', async () => {
    function* dataGenerator() {
      const numElements = 5;
      let index = 0;
      while (index < numElements) {
        const x = index;
        index++;
        yield x;
      }
    }
    const ds = tfd.generator(dataGenerator);
    const result = await ds.toArrayForTest();
    expect(result).toEqual([0, 1, 2, 3, 4]);
  });

  it('generate multiple datasets from JavaScript generator', async () => {
    function* dataGenerator() {
      const numElements = 5;
      let index = 0;
      while (index < numElements) {
        const x = index;
        index++;
        yield x;
      }
    }
    const ds = tfd.generator(dataGenerator);
    const result1 = await ds.toArrayForTest();
    expect(result1).toEqual([0, 1, 2, 3, 4]);
    const result2 = await ds.toArrayForTest();
    expect(result2).toEqual([0, 1, 2, 3, 4]);
  });

  it('generate dataset from JavaScript iterator factory', async () => {
    function makeIterator() {
      let iterationCount = 0;
      const iterator = {
        next: () => {
          let result;
          if (iterationCount < 5) {
            result = {value: iterationCount, done: false};
            iterationCount++;
            return result;
          }
          return {value: iterationCount, done: true};
        }
      };
      return iterator;
    }
    const ds = tfd.generator(makeIterator);
    const result = await ds.toArrayForTest();
    expect(result).toEqual([0, 1, 2, 3, 4]);
  });

  it('generate multiple datasets from JavaScript iterator factory',
     async () => {
       function makeIterator() {
         let iterationCount = 0;
         const iterator = {
           next: () => {
             let result;
             if (iterationCount < 5) {
               result = {value: iterationCount, done: false};
               iterationCount++;
               return result;
             }
             return {value: iterationCount, done: true};
           }
         };
         return iterator;
       }
       const ds = tfd.generator(makeIterator);
       const result1 = await ds.toArrayForTest();
       expect(result1).toEqual([0, 1, 2, 3, 4]);
       const result2 = await ds.toArrayForTest();
       expect(result2).toEqual([0, 1, 2, 3, 4]);
     });

  it('generate dataset from async iterator factory', async () => {
    async function waitAndCreateCount() {
      return new Promise(resolve => {
        setTimeout(() => {
          resolve(3);
        }, 1000);
      });
    }
    async function makeIterator() {
      let iterationCount = (await waitAndCreateCount()) as number;
      const iterator = {
        next: () => {
          let result;
          if (iterationCount < 6) {
            result = {value: iterationCount, done: false};
            iterationCount++;
            return result;
          }
          return {value: iterationCount, done: true};
        }
      };
      return iterator;
    }
    const ds = tfd.generator(makeIterator);
    const result = await ds.toArrayForTest();
    expect(result).toEqual([3, 4, 5]);
  });
});
