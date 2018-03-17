/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit tests for executor_test.ts.
 */

// tslint:disable:max-line-length
import {Tensor, tensor1d, tensor2d, tensor3d} from '@tensorflow/tfjs-core';

import * as K from '../backend/deeplearnjs_backend';
import {Dense} from '../layers/core';
import {DType} from '../types';
import {SymbolicTensor} from '../types';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {execute, FeedDict} from './executor';
import {Input} from './topology';

// tslint:enable

describeMathCPU('FeedDict', () => {
  const x = Input({shape: [], name: 'x', dtype: DType.float32});
  const y = Input({shape: [], name: 'y', dtype: DType.float32});
  const xValue = tensor1d([42]);
  const yValue = tensor1d([21]);

  it('FeedDict from a single Feed', () => {
    const feedDict = new FeedDict([{key: x, value: xValue}]);

    expect(feedDict.hasKey(x)).toBe(true);
    expect(feedDict.hasKey(y)).toBe(false);
    expect(feedDict.getValue(x)).toEqual(xValue);
    expect(() => feedDict.getValue(y)).toThrowError();
  });
  it('FeedDict from duplicate Feeds throws error', () => {
    const feed = {key: x, value: xValue};
    expect(() => new FeedDict([feed, feed])).toThrowError(/Duplicate key/);
  });
  it('Add key and value', () => {
    const feedDict = new FeedDict();
    expect(feedDict.hasKey(x)).toBe(false);
    expect(feedDict.hasKey(y)).toBe(false);

    expect(feedDict.add(x, xValue)).toEqual(feedDict);
    expect(feedDict.hasKey(x)).toBe(true);
    expect(feedDict.hasKey(y)).toBe(false);

    expect(feedDict.add(y, yValue)).toEqual(feedDict);
    expect(feedDict.hasKey(x)).toBe(true);
    expect(feedDict.hasKey(y)).toBe(true);
    expect(feedDict.getValue(x)).toEqual(xValue);
    expect(feedDict.getValue(y)).toEqual(yValue);
  });
  it('Copy constructor', () => {
    const feedDict1 = new FeedDict().add(x, xValue);
    const feedDict2 = new FeedDict(feedDict1);
    expect(feedDict2.hasKey(x)).toBe(true);
    expect(feedDict2.getValue(x)).toEqual(xValue);
    expect(feedDict2.hasKey(y)).toBe(false);

    feedDict2.add(y, yValue);
    expect(feedDict2.hasKey(y)).toBe(true);
    expect(feedDict2.getValue(y)).toEqual(yValue);
    expect(feedDict1.hasKey(y)).toBe(false);
  });
  it('Add duplicate key and value leads to error', () => {
    const feedDict = new FeedDict();

    expect(feedDict.add(x, xValue)).toEqual(feedDict);
    expect(() => feedDict.add(x, xValue)).toThrowError(/Duplicate key/);
  });
  it('Feeding compatible value with undetermined dimension works', () => {
    const s = Input({shape: [null, 4], name: 's', dtype: DType.float32});
    const sValue = tensor3d([1, 3, 3, 7], [1, 1, 4]);
    const feedDict = new FeedDict([{key: s, value: sValue}]);
    expect(feedDict.getValue(s)).toEqual(sValue);
  });
  it('Feeding incompatible rank leads to error', () => {
    const s = Input({shape: [null, 4], name: 's', dtype: DType.float32});
    const sValue = tensor2d([1, 3, 3, 7], [1, 4]);
    expect(() => new FeedDict([{key: s, value: sValue}]))
        .toThrowError(/rank of feed .* does not match/);
  });
  it('Feeding incompatible dimension leads to error', () => {
    const s = Input({shape: [null, 4], name: 's', dtype: DType.float32});
    const sValue = tensor3d([0, 0, 8], [1, 1, 3]);
    expect(() => new FeedDict([{key: s, value: sValue}]))
        .toThrowError(/The 2-th dimension of the feed .* is incompatible/);
  });
});

describeMathCPUAndGPU('Executor', () => {
  it('Linear Graph Topology', () => {
    const x = Input({shape: [2], name: 'fooInput', dtype: DType.float32});
    const denseLayer1 =
        new Dense({units: 5, activation: 'linear', kernelInitializer: 'Ones'});
    const y = denseLayer1.apply(x);
    const u = Input({shape: [2], name: 'fooInput', dtype: DType.float32});
    const denseLayer2 =
        new Dense({units: 5, activation: 'linear', kernelInitializer: 'Ones'});
    const denseLayer3 =
        new Dense({units: 3, activation: 'linear', kernelInitializer: 'Ones'});
    const v = denseLayer2.apply(u);
    const w = denseLayer3.apply(v);

    it('Execute Input directly', () => {
      const xValue = K.ones([2, 2]);
      const feedDict = new FeedDict().add(x, xValue);
      expectTensorsClose(
          execute(x as SymbolicTensor, feedDict) as Tensor,
          tensor2d([1, 1, 1, 1], [2, 2]));
    });
    it('Input to Dense', () => {
      const xValue = K.ones([2, 2]);
      const feedDict = new FeedDict([{key: x, value: xValue}]);
      expectTensorsClose(
          execute(y as SymbolicTensor, feedDict) as Tensor,
          tensor2d([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 5]));
    });
    it('Input to Dense1 to Dense2', () => {
      const uValue = K.ones([2, 2]);
      const feedDict = new FeedDict([{key: u, value: uValue}]);
      expectTensorsClose(
          execute(w as SymbolicTensor, feedDict) as Tensor,
          tensor2d([10, 10, 10, 10, 10, 10], [2, 3]));
    });
    it('Feed value to intermediate layers is supported', () => {
      const vValue = K.ones([3, 5]);
      const feedDict =
          new FeedDict([{key: v as SymbolicTensor, value: vValue}]);
      expectTensorsClose(
          execute(w as SymbolicTensor, feedDict) as Tensor,
          tensor2d([5, 5, 5, 5, 5, 5, 5, 5, 5], [3, 3]));
    });
    it('Calling execute without all input feeds available leads to error',
       () => {
         const feedDict = new FeedDict();
         expect(() => execute(y as SymbolicTensor, feedDict))
             .toThrowError(/Missing a feed value .* from InputLayer/);
       });
  });

  it('Diamond Graph Topology', () => {
    const x = Input({shape: [2], name: 'fooInput', dtype: DType.float32});
    const denseLayer1 = new Dense({
      units: 5,
      activation: 'linear',
      kernelInitializer: 'Ones',
      name: 'denseLayer1'
    });
    const y = denseLayer1.apply(x);
    const denseLayer2 = new Dense({
      units: 4,
      activation: 'linear',
      kernelInitializer: 'Ones',
      name: 'denseLayer2'
    });
    const denseLayer3 = new Dense({
      units: 3,
      activation: 'linear',
      kernelInitializer: 'Ones',
      name: 'denseLayer3'
    });
    const z1 = denseLayer2.apply(y) as SymbolicTensor;
    const z2 = denseLayer3.apply(y) as SymbolicTensor;

    it('Calling execute with two fetches and diamond graph works', () => {
      const xValue = K.ones([2, 2]);
      const feedDict = new FeedDict([{key: x, value: xValue}]);
      let callCounter = 0;
      denseLayer1.setCallHook(() => {
        callCounter++;
      });

      const outputs = execute([z1, z2], feedDict) as Tensor[];
      expectTensorsClose(
          outputs[0], tensor2d([10, 10, 10, 10, 10, 10, 10, 10], [2, 4]));
      expectTensorsClose(
          outputs[1], tensor2d([2, 3], [10, 10, 10, 10, 10, 10]));
      // The counter should have been incremented twice, because execute() is
      // called twice, once on CPU and once on GPU.
      expect(callCounter).toEqual(2);
    });
  });
});
