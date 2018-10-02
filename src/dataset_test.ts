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
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
import {TensorContainerObject} from '@tensorflow/tfjs-core/dist/tensor_types';

import {Dataset, datasetFromElements, datasetFromIteratorFn, zip} from './dataset';
import {iteratorFromFunction, iteratorFromItems, LazyIterator} from './iterators/lazy_iterator';
import {DataElementObject, DatasetContainer} from './types';

class TestObjectIterator extends LazyIterator<{}> {
  data = Array.from({length: 100}, (v, k) => k);
  currentIndex = 0;

  summary() {
    return `TestObjects`;
  }

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

export class TestDataset extends Dataset<DataElementObject> {
  async iterator(): Promise<LazyIterator<{}>> {
    return new TestObjectIterator();
  }
}

// tslint:disable-next-line:no-any
function complexifyExample(simple: any): {} {
  const v = simple['number'];
  const w = simple['numberArray'];
  const x = simple['Tensor'];
  const y = simple['Tensor2'];
  const z = simple['string'];
  return {
    a: [v, w, {aa: [x, y, z], ab: [v, w, x]}],
    b: {
      ba: {baa: y, bab: z, bac: v},
      bb: {bba: w, bbb: x, bbc: y},
      bc: {bca: z, bcb: v, bcc: w}
    },
    c: [[x, y, z], [v, w, x], [y, z, v]]
  };
}

describeWithFlags('Dataset', tf.test_util.CPU_ENVS, () => {
  it('can be concatenated', async () => {
    const a = datasetFromElements([{'item': 1}, {'item': 2}, {'item': 3}]);
    const b = datasetFromElements([{'item': 4}, {'item': 5}, {'item': 6}]);
    const result = await a.concatenate(b).collectAll();
    expect(result).toEqual([
      {'item': 1}, {'item': 2}, {'item': 3}, {'item': 4}, {'item': 5},
      {'item': 6}
    ]);
  });

  it('can be created by concatenating multiple underlying datasets via reduce',
     async () => {
       const a = datasetFromElements([{'item': 1}, {'item': 2}]);
       const b = datasetFromElements([{'item': 3}, {'item': 4}]);
       const c = datasetFromElements([{'item': 5}, {'item': 6}]);
       const concatenated = [a, b, c].reduce((a, b) => a.concatenate(b));
       const result = await concatenated.collectAll();
       expect(result).toEqual([
         {'item': 1}, {'item': 2}, {'item': 3}, {'item': 4}, {'item': 5},
         {'item': 6}
       ]);
     });

  it('can be created by zipping an array of datasets with primitive elements',
     async () => {
       const a = datasetFromElements([1, 2, 3]);
       const b = datasetFromElements([4, 5, 6]);
       const result = await zip([a, b]).collectAll();
       expect(result).toEqual([[1, 4], [2, 5], [3, 6]]);
     });

  it('can be created by zipping an array of datasets with object elements',
     async () => {
       const a = datasetFromElements([{a: 1}, {a: 2}, {a: 3}]);
       const b = datasetFromElements([{b: 4}, {b: 5}, {b: 6}]);
       const result = await zip([a, b]).collectAll();
       expect(result).toEqual(
           [[{a: 1}, {b: 4}], [{a: 2}, {b: 5}], [{a: 3}, {b: 6}]]);
     });

  it('can be created by zipping a dict of datasets', async () => {
    const a = datasetFromElements([{a: 1}, {a: 2}, {a: 3}]);
    const b = datasetFromElements([{b: 4}, {b: 5}, {b: 6}]);
    const result = await zip({c: a, d: b}).collectAll();
    expect(result).toEqual([
      {c: {a: 1}, d: {b: 4}}, {c: {a: 2}, d: {b: 5}}, {c: {a: 3}, d: {b: 6}}
    ]);
  });

  it('can be created by zipping a nested structure of datasets', async () => {
    const a = datasetFromElements([1, 2, 3]);
    const b = datasetFromElements([4, 5, 6]);
    const c = datasetFromElements([7, 8, 9]);
    const d = datasetFromElements([10, 11, 12]);
    const result = await zip({a, bcd: [b, {c, d}]}).collectAll();

    expect(result).toEqual([
      {a: 1, bcd: [4, {c: 7, d: 10}]},
      {a: 2, bcd: [5, {c: 8, d: 11}]},
      {a: 3, bcd: [6, {c: 9, d: 12}]},
    ]);
  });

  it('can be created by zipping datasets of different sizes', async () => {
    const a = datasetFromElements([1, 2]);
    const b = datasetFromElements([3, 4, 5, 6]);
    const result = await zip([a, b]).collectAll();
    expect(result).toEqual([[1, 3], [2, 4]]);
  });

  it('zipping a native string throws an error', async done => {
    try {
      // tslint:disable-next-line:no-any no-construct
      await zip('test' as any);
      done.fail();
    } catch (e) {
      expect(e.message).toEqual(
          'The argument to zip() must be an object or array.');
      done();
    }
  });

  it('zipping a string object throws a meaningful error', async done => {
    try {
      // tslint:disable-next-line:no-any no-construct
      await zip(new String('test') as any).iterator();
      done.fail();
    } catch (e) {
      // This error is not specific to the error case arising from
      //   typeof(new String('test')) === 'object'
      // Instead this error is thrown because the leaves of the structure are
      // the letters t, e, s, and t, as well a number for the length.
      // I think it's a fine error message for this situation anyway.
      expect(e.message).toEqual(
          'Leaves of the structure passed to zip() must be Datasets, ' +
          'not primitives.');
      done();
    }
  });

  it('zipping a structure with repeated elements works', async () => {
    const a = datasetFromElements([1, 2, 3]);
    const b = datasetFromElements([4, 5, 6]);
    const c = datasetFromElements([7, 8, 9]);
    const d = datasetFromElements([10, 11, 12]);
    const result = await zip({a, abacd: [a, b, {a, c, d}]}).collectAll();

    expect(result).toEqual([
      {a: 1, abacd: [1, 4, {a: 1, c: 7, d: 10}]},
      {a: 2, abacd: [2, 5, {a: 2, c: 8, d: 11}]},
      {a: 3, abacd: [3, 6, {a: 3, c: 9, d: 12}]},
    ]);
  });

  it('zipping a structure with cycles throws an error', async done => {
    try {
      // tslint:disable-next-line:no-any
      const a = datasetFromElements([1, 2, 3]);
      const b = datasetFromElements([4, 5, 6]);
      const c: DatasetContainer = [datasetFromElements([7, 8, 9])];
      const abc: DatasetContainer = [a, b, c];
      c.push(abc);
      await zip({a, abc}).iterator();
      done.fail();
    } catch (e) {
      expect(e.message).toEqual('Circular references are not supported.');
      done();
    }
  });

  it('zip propagates errors thrown when iterating constituent datasets',
     async done => {
       try {
         let count = 0;
         const a =
             datasetFromIteratorFn(async () => iteratorFromFunction(() => {
                                     if (count > 2) {
                                       throw new Error('propagate me!');
                                     }
                                     return {value: count++, done: false};
                                   }));
         const b = datasetFromElements([3, 4, 5, 6]);
         // tslint:disable-next-line:no-any
         await (await zip([a, b]).iterator()).collect(1000, 0);
         done.fail();
       } catch (e) {
         expect(e.message).toEqual(
             'Error thrown while iterating through a dataset: propagate me!');
         done();
       }
     });

  it('can be repeated a fixed number of times', async () => {
    const a = datasetFromElements([{'item': 1}, {'item': 2}, {'item': 3}]);
    const result = await a.repeat(4).collectAll();
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
  });

  it('can be repeated indefinitely', async () => {
    const a = datasetFromElements([{'item': 1}, {'item': 2}, {'item': 3}]);
    await a.repeat().take(234).collectAll();
  });

  it('can be repeated with state in a closure', async () => {
    // This tests a tricky bug having to do with 'this' being set properly.
    // See
    // https://github.com/Microsoft/TypeScript/wiki/%27this%27-in-TypeScript

    class CustomDataset extends Dataset<{}> {
      state = {val: 1};
      async iterator() {
        const result = iteratorFromItems([
          {'item': this.state.val++}, {'item': this.state.val++},
          {'item': this.state.val++}
        ]);
        return result;
      }
    }
    const a = new CustomDataset();
    await a.repeat().take(1234).collectAll();
  });

  it('can collect all items into memory', async () => {
    const ds = new TestDataset();
    const items = await ds.collectAll();
    expect(items.length).toEqual(100);
    // The test dataset has 100 elements, each containing 2 Tensors.
    expect(tf.memory().numTensors).toEqual(200);
  });

  it('batches entries into column-oriented batches', async () => {
    const ds = new TestDataset();
    const bds = ds.batch(8);
    const batchIterator = await bds.iterator();
    const result = await batchIterator.collect();

    expect(result.length).toEqual(13);
    result.slice(0, 12).forEach(batch => {
      const b = batch as TensorContainerObject;
      expect((b['number'] as tf.Tensor).shape).toEqual([8]);
      expect((b['numberArray'] as tf.Tensor).shape).toEqual([8, 3]);
      expect((b['Tensor'] as tf.Tensor).shape).toEqual([8, 3]);
      expect((b['Tensor2'] as tf.Tensor).shape).toEqual([8, 2, 2]);
      expect((b['string'] as string[]).length).toEqual(8);
    });
    tf.dispose(result);
    expect(tf.ENV.engine.memory().numTensors).toBe(0);
  });

  it('batches entries without leaking Tensors', async () => {
    // The prior test confirms this too, but this formulation is more specific.

    // First show that an unbatched test iterator creates 2 Tensors per element.
    const ds = new TestDataset();
    const iter = await ds.iterator();
    const element = await iter.next();
    expect(tf.ENV.engine.memory().numTensors).toBe(2);
    tf.dispose(element.value);
    expect(tf.ENV.engine.memory().numTensors).toBe(0);

    // Now obtain batches, which should contain 4 Tensors each.
    const bDs = new TestDataset().batch(8);
    const bIter = await bDs.iterator();
    const bElement = await bIter.next();
    // The batch element contains four Tensors, and the 8*2 Tensors in the
    // original unbatched elements have already been disposed.
    expect(tf.ENV.engine.memory().numTensors).toBe(4);
    tf.dispose(bElement.value);
    expect(tf.ENV.engine.memory().numTensors).toBe(0);
  });

  it('batches complex nested entries into column-oriented batches',
     async () => {
       // Our "complexified" examples map the simple examples into a deep
       // nested structure. This test shows that batching complexified examples
       // produces the same result as complexifying batches of simple examples.

       const complexThenBatch =
           new TestDataset().map(complexifyExample).batch(8);
       const batchThenComplex =
           new TestDataset().batch(8).map(complexifyExample);

       const compareDataset = zip({complexThenBatch, batchThenComplex});

       const result = await (await compareDataset.iterator()).collect();

       expect(result.length).toEqual(13);
       // tslint:disable-next-line:no-any
       result.slice(0, 12).forEach((compare: any) => {
         // TODO(soergel): could use deepMap to implement deep compare.
         // For now, just spot-check a few deep entries.
         expect(compare.complexThenBatch.a[0].shape)
             .toEqual(compare.batchThenComplex.a[0].shape);
         expect(compare.complexThenBatch.a[0].dataSync())
             .toEqual(compare.batchThenComplex.a[0].dataSync());

         expect(compare.complexThenBatch.a[2].ab[2].shape)
             .toEqual(compare.batchThenComplex.a[2].ab[2].shape);
         expect(compare.complexThenBatch.a[2].ab[2].dataSync())
             .toEqual(compare.batchThenComplex.a[2].ab[2].dataSync());

         expect(compare.complexThenBatch.b.bb.bbb.shape)
             .toEqual(compare.batchThenComplex.b.bb.bbb.shape);
         expect(compare.complexThenBatch.b.bb.bbb.dataSync())
             .toEqual(compare.batchThenComplex.b.bb.bbb.dataSync());

         expect(compare.complexThenBatch.c[1][1].shape)
             .toEqual(compare.batchThenComplex.c[1][1].shape);
         expect(compare.complexThenBatch.c[1][1].dataSync())
             .toEqual(compare.batchThenComplex.c[1][1].dataSync());
       });
       tf.dispose(result);
       expect(tf.ENV.engine.memory().numTensors).toBe(0);
     });

  it('batch creates a small last batch', async () => {
    const ds = new TestDataset();
    const bds = ds.batch(8);
    const batchIterator = await bds.iterator();
    const result = await batchIterator.collect();
    const lastBatch = result[result.length - 1] as TensorContainerObject;
    expect((lastBatch['number'] as tf.Tensor).shape).toEqual([4]);
    expect((lastBatch['numberArray'] as tf.Tensor).shape).toEqual([4, 3]);
    expect((lastBatch['Tensor'] as tf.Tensor).shape).toEqual([4, 3]);
    expect((lastBatch['Tensor2'] as tf.Tensor).shape).toEqual([4, 2, 2]);
    expect((lastBatch['string'] as string[]).length).toEqual(4);

    const expectedNumberLastBatch = tf.tensor1d([96, 97, 98, 99]);
    tf.test_util.expectArraysClose(
        lastBatch['number'] as tf.Tensor, expectedNumberLastBatch);

    const expectedNumberArrayLastBatch = tf.tensor2d(
        [
          [96, 96 ** 2, 96 ** 3], [97, 97 ** 2, 97 ** 3],
          [98, 98 ** 2, 98 ** 3], [99, 99 ** 2, 99 ** 3]
        ],
        [4, 3]);
    tf.test_util.expectArraysClose(
        lastBatch['numberArray'] as tf.Tensor, expectedNumberArrayLastBatch);

    const expectedTensorLastBatch = tf.tensor2d(
        [
          [96, 96 ** 2, 96 ** 3], [97, 97 ** 2, 97 ** 3],
          [98, 98 ** 2, 98 ** 3], [99, 99 ** 2, 99 ** 3]
        ],
        [4, 3]);
    tf.test_util.expectArraysClose(
        lastBatch['Tensor'] as tf.Tensor, expectedTensorLastBatch);

    const expectedTensor2LastBatch = tf.tensor3d(
        [
          [[96, 96 ** 2], [96 ** 3, 96 ** 4]],
          [[97, 97 ** 2], [97 ** 3, 97 ** 4]],
          [[98, 98 ** 2], [98 ** 3, 98 ** 4]],
          [[99, 99 ** 2], [99 ** 3, 99 ** 4]],
        ],
        [4, 2, 2]);
    tf.test_util.expectArraysClose(
        lastBatch['Tensor2'] as tf.Tensor, expectedTensor2LastBatch);

    const expectedStringLastBatch =
        ['Item 96', 'Item 97', 'Item 98', 'Item 99'];
    expect(lastBatch['string'] as string[]).toEqual(expectedStringLastBatch);

    tf.dispose(result);
    tf.dispose(expectedNumberLastBatch);
    tf.dispose(expectedNumberArrayLastBatch);
    tf.dispose(expectedTensorLastBatch);
    tf.dispose(expectedTensor2LastBatch);

    expect(tf.ENV.engine.memory().numTensors).toBe(0);
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

  it('filter does not leak Tensors', async () => {
    const ds = new TestDataset();
    expect(tf.memory().numTensors).toEqual(0);
    await ds.filter(x => ((x['number'] as number) % 2 === 0)).collectAll();
    // Each element of the test dataset contains 2 Tensors.
    // There were 100 elements, but we filtered out half of them.
    // Thus 50 * 2 = 100 Tensors remain.
    expect(tf.memory().numTensors).toEqual(100);
  });

  it('map does not leak Tensors when none are returned', async () => {
    const ds = new TestDataset();
    expect(tf.memory().numTensors).toEqual(0);
    await ds.map(x => ({'constant': 1})).collectAll();
    // The map operation consumed all of the tensors and emitted none.
    expect(tf.memory().numTensors).toEqual(0);
  });

  it('map does not lose or leak Tensors when some inputs are passed through',
     async () => {
       const ds = new TestDataset();
       expect(tf.memory().numTensors).toEqual(0);
       await ds.map(x => ({'Tensor2': x['Tensor2']})).collectAll();
       // Each element of the test dataset contains 2 Tensors.
       // Our map operation retained one of the Tensors and discarded the
       // other. Thus the mapped data contains 100 elements with 1 Tensor
       // each.
       expect(tf.memory().numTensors).toEqual(100);
     });

  it('map does not leak Tensors when inputs are replaced', async () => {
    const ds = new TestDataset();
    expect(tf.memory().numTensors).toEqual(0);
    await ds.map(x => ({'a': tf.tensor1d([1, 2, 3])})).collectAll();
    // Each element of the test dataset contains 2 Tensors.
    // Our map operation discarded both Tensors and created one new one.
    // Thus the mapped data contains 100 elements with 1 Tensor each.
    expect(tf.memory().numTensors).toEqual(100);
  });

  it('forEach does not leak Tensors', async () => {
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
  });
});
