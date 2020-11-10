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

import {dispose, memory, ones, Tensor, tensor1d, tensor2d, tensor3d, zeros} from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {execute, ExecutionProbe, FeedDict, getTopologicalSortAndRecipientCountsForOneFetch} from './executor';

// tslint:enable

describeMathCPU('FeedDict', () => {
  const x = tfl.input({shape: [], name: 'x', dtype: 'float32'});
  const y = tfl.input({shape: [], name: 'y', dtype: 'float32'});
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
  it('getValue by tensor name', () => {
    const feedDict = new FeedDict();
    expect(feedDict.add(x, xValue)).toEqual(feedDict);
    expect(feedDict.add(y, yValue)).toEqual(feedDict);

    expect(feedDict.getValue(x.name)).toEqual(xValue);
    expect(feedDict.getValue(y.name)).toEqual(yValue);
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
    const s = tfl.input({shape: [null, 4], name: 's', dtype: 'float32'});
    const sValue = tensor3d([1, 3, 3, 7], [1, 1, 4]);
    const feedDict = new FeedDict([{key: s, value: sValue}]);
    expect(feedDict.getValue(s)).toEqual(sValue);
  });
});

describeMathCPU('getTopologicalSortAndRecipientCountsForOneFetch', () => {
  it('Triangular topology', () => {
    const input = tfl.input({shape: [2, 6]});
    const f1 = tfl.layers.flatten().apply(input) as tfl.SymbolicTensor;
    const r1 = tfl.layers.reLU().apply(f1) as tfl.SymbolicTensor;
    const c1 = tfl.layers.concatenate().apply([f1, r1]) as tfl.SymbolicTensor;
    const relu2 = tfl.layers.reLU().apply(c1) as tfl.SymbolicTensor;

    const {sorted, recipientMap} =
        getTopologicalSortAndRecipientCountsForOneFetch(relu2, new FeedDict());
    expect(sorted).toEqual([input, f1, r1, c1, relu2]);
    expect(recipientMap[input.name].size).toEqual(1);
    expect(recipientMap[f1.name].size).toEqual(2);
    expect(recipientMap[r1.name].size).toEqual(1);
    expect(recipientMap[c1.name].size).toEqual(1);
  });

  it('Double triangular topology', () => {
    const input = tfl.input({shape: [2, 6]});
    const f1 = tfl.layers.flatten().apply(input) as tfl.SymbolicTensor;
    const r1 = tfl.layers.reLU().apply(f1) as tfl.SymbolicTensor;
    const c1 = tfl.layers.concatenate().apply([f1, r1]) as tfl.SymbolicTensor;
    const r2 = tfl.layers.reLU().apply(c1) as tfl.SymbolicTensor;
    const c2 = tfl.layers.concatenate().apply([f1, r2]) as tfl.SymbolicTensor;
    const r3 = tfl.layers.reLU().apply(c2) as tfl.SymbolicTensor;

    const {sorted, recipientMap} =
        getTopologicalSortAndRecipientCountsForOneFetch(r3, new FeedDict());
    expect(sorted).toEqual([input, f1, r1, c1, r2, c2, r3]);
    expect(recipientMap[input.name].size).toEqual(1);
    expect(recipientMap[f1.name].size).toEqual(3);
    expect(recipientMap[r1.name].size).toEqual(1);
    expect(recipientMap[c1.name].size).toEqual(1);
    expect(recipientMap[r2.name].size).toEqual(1);
    expect(recipientMap[c2.name].size).toEqual(1);
  });
});

describeMathCPUAndGPU('Executor', () => {
  describe('Linear Graph Topology', () => {
    let x: tfl.SymbolicTensor;
    let y: {};
    let u: tfl.SymbolicTensor;
    let v: {};
    let w: {};

    beforeEach(() => {
      x = tfl.input({shape: [2], name: 'fooInput', dtype: 'float32'});
      const denseLayer1 = tfl.layers.dense(
          {units: 5, activation: 'linear', kernelInitializer: 'ones'});
      y = denseLayer1.apply(x);
      u = tfl.input({shape: [2], name: 'footInput', dtype: 'float32'});
      const denseLayer2 = tfl.layers.dense(
          {units: 5, activation: 'linear', kernelInitializer: 'ones'});
      const denseLayer3 = tfl.layers.dense(
          {units: 3, activation: 'linear', kernelInitializer: 'ones'});
      v = denseLayer2.apply(u);
      w = denseLayer3.apply(v as tfl.SymbolicTensor);
    });

    it('Execute Input directly', () => {
      const xValue = ones([2, 2]);
      const feedDict = new FeedDict().add(x, xValue);
      expectTensorsClose(
          execute(x, feedDict) as Tensor, tensor2d([1, 1, 1, 1], [2, 2]));
    });
    it('Input to Dense', () => {
      const xValue = ones([2, 2]);
      const feedDict = new FeedDict([{key: x, value: xValue}]);
      expectTensorsClose(
          execute(y as tfl.SymbolicTensor, feedDict) as Tensor,
          tensor2d([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 5]));
    });
    it('Input to Dense1 to Dense2', () => {
      const uValue = ones([2, 2]);
      const feedDict = new FeedDict([{key: u, value: uValue}]);
      expectTensorsClose(
          execute(w as tfl.SymbolicTensor, feedDict) as Tensor,
          tensor2d([10, 10, 10, 10, 10, 10], [2, 3]));
    });
    it('Feed value to intermediate layers is supported', () => {
      const vValue = ones([3, 5]);
      const feedDict =
          new FeedDict([{key: v as tfl.SymbolicTensor, value: vValue}]);
      expectTensorsClose(
          execute(w as tfl.SymbolicTensor, feedDict) as Tensor,
          tensor2d([5, 5, 5, 5, 5, 5, 5, 5, 5], [3, 3]));
    });
    it('Calling execute without all Input feeds available leads to error',
       () => {
         const feedDict = new FeedDict();
         expect(() => execute(y as tfl.SymbolicTensor, feedDict)).toThrow();
       });

    it('Maximum memory use under linear graph topology', () => {
      const input = tfl.input({shape: [2, 3]});
      let y: tfl.SymbolicTensor = input;
      for (let i = 0; i < 10; ++i) {
        y = tfl.layers.reshape({targetShape:　i % 2 === 0 ? [6] : [3, 2]})
                .apply(y) as tfl.SymbolicTensor;
      }
      const feedDict = new FeedDict([{key: input, value: zeros([4, 2, 3])}]);
      const numTensors0 = memory().numTensors;
      const probe: ExecutionProbe = {};
      dispose(execute(y, feedDict, null, probe));
      // Assert no memory leak.
      expect(memory().numTensors).toEqual(numTensors0);
      // Assert that intermediate tensors are cleaned up properly during
      // execution.
      expect(probe.maxNumTensors).toBeLessThanOrEqual(numTensors0 + 1);
    });
  });

  describe('Diamond Graph Topology', () => {
    it('Calling execute with two fetches and diamond graph works', () => {
      const x = tfl.input({shape: [2], name: 'fooInput', dtype: 'float32'});
      const denseLayer1 = tfl.layers.dense({
        units: 5,
        activation: 'linear',
        kernelInitializer: 'ones',
        name: 'denseLayer1'
      });
      const y = denseLayer1.apply(x);
      const denseLayer2 = tfl.layers.dense({
        units: 4,
        activation: 'linear',
        kernelInitializer: 'ones',
        name: 'denseLayer2'
      });
      const denseLayer3 = tfl.layers.dense({
        units: 3,
        activation: 'linear',
        kernelInitializer: 'ones',
        name: 'denseLayer3'
      });
      const z1 = denseLayer2.apply(y) as tfl.SymbolicTensor;
      const z2 = denseLayer3.apply(y) as tfl.SymbolicTensor;

      const xValue = ones([2, 2]);
      const feedDict = new FeedDict([{key: x, value: xValue}]);
      let callCounter = 0;
      denseLayer1.setCallHook(() => {
        callCounter++;
      });

      const outputs = execute([z1, z2], feedDict) as Tensor[];
      expectTensorsClose(
          outputs[0], tensor2d([10, 10, 10, 10, 10, 10, 10, 10], [2, 4]));
      expectTensorsClose(
          outputs[1], tensor2d([10, 10, 10, 10, 10, 10], [2, 3]));
      expect(callCounter).toEqual(1);
    });
  });
});
