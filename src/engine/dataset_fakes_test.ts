/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {Tensor} from '@tensorflow/tfjs-core';

import {describeMathCPUAndGPU} from '../utils/test_utils';
import {FakeNumericDataset} from './dataset_fakes';
import {TensorMap} from './dataset_stub';

describeMathCPUAndGPU('FakeNumericDataset', () => {
  it('1D features, 1D targets', async () => {
    const dataset = new FakeNumericDataset(
        {xShape: [3], yShape: [1], batchSize: 8, numBatches: 5});
    for (let k = 0; k < 2; ++k) {
      // Run twice to make sure that calling iteartor() multiple times works.
      const iterator = await dataset.iterator();
      for (let i = 0; i < 5; ++i) {
        const result = await iterator.next();
        expect(result.value.length).toEqual(2);
        expect((result.value[0] as Tensor).shape).toEqual([8, 3]);
        expect((result.value[1] as Tensor).shape).toEqual([8, 1]);
        expect(result.done).toEqual(false);
      }
      for (let i = 0; i < 3; ++i) {
        const result = await iterator.next();
        expect(result.value).toBeNull();
        expect(result.done).toEqual(true);
      }
    }
  });

  it('2D features, 1D targets', async () => {
    const dataset = new FakeNumericDataset(
        {xShape: [3, 4], yShape: [2], batchSize: 8, numBatches: 5});
    for (let k = 0; k < 2; ++k) {
      // Run twice to make sure that calling iteartor() multiple times works.
      const iterator = await dataset.iterator();
      for (let i = 0; i < 5; ++i) {
        const result = await iterator.next();
        expect(result.value.length).toEqual(2);
        expect((result.value[0] as Tensor).shape).toEqual([8, 3, 4]);
        expect((result.value[1] as Tensor).shape).toEqual([8, 2]);
        expect(result.done).toEqual(false);
      }
      for (let i = 0; i < 3; ++i) {
        const result = await iterator.next();
        expect(result.value).toBeNull();
        expect(result.done).toEqual(true);
      }
    }
  });

  it('Multiple 2D features, 1D targets', async () => {
    const dataset = new FakeNumericDataset({
      xShape: {'input1': [3, 4], 'input2': [2, 3]},
      yShape: [2],
      batchSize: 8,
      numBatches: 5
    });
    for (let k = 0; k < 2; ++k) {
      // Run twice to make sure that calling iteartor() multiple times works.
      const iterator = await dataset.iterator();
      for (let i = 0; i < 5; ++i) {
        const result = await iterator.next();
        expect(result.value.length).toEqual(2);
        const xs = result.value[0] as TensorMap;
        expect(xs['input1'].shape).toEqual([8, 3, 4]);
        expect(xs['input2'].shape).toEqual([8, 2, 3]);
        expect((result.value[1] as Tensor).shape).toEqual([8, 2]);
        expect(result.done).toEqual(false);
      }
      for (let i = 0; i < 3; ++i) {
        const result = await iterator.next();
        expect(result.value).toBeNull();
        expect(result.done).toEqual(true);
      }
    }
  });

  it('Invalid batchSize leads to Error', () => {
    expect(
        () => new FakeNumericDataset(
            {xShape: [3], yShape: [1], batchSize: -8, numBatches: 5}))
        .toThrow();
    expect(
        () => new FakeNumericDataset(
            {xShape: [3], yShape: [1], batchSize: 8.5, numBatches: 5}))
        .toThrow();
    expect(
        () => new FakeNumericDataset(
            {xShape: [3], yShape: [1], batchSize: 0, numBatches: 5}))
        .toThrow();
    expect(
        () => new FakeNumericDataset(
            // tslint:disable-next-line:no-any
            {xShape: [3], yShape: [1], batchSize: 'foo' as any, numBatches: 5}))
        .toThrow();
  });

  it('Invalid numBatches leads to Error', () => {
    expect(
        () => new FakeNumericDataset(
            {xShape: [3], yShape: [1], batchSize: 8, numBatches: -5}))
        .toThrow();
    expect(
        () => new FakeNumericDataset(
            {xShape: [3], yShape: [1], batchSize: 8, numBatches: 5.5}))
        .toThrow();
    expect(
        () => new FakeNumericDataset(
            {xShape: [3], yShape: [1], batchSize: 8, numBatches: 0}))
        .toThrow();
    expect(
        () => new FakeNumericDataset(
            // tslint:disable-next-line:no-any
            {xShape: [3], yShape: [1], batchSize: 8, numBatches: 'foo' as any}))
        .toThrow();
  });
});
