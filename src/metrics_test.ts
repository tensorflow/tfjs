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
 * Unit tests for metrics.ts.
 */

import {scalar, tensor1d, tensor2d} from 'deeplearn';

import {binaryAccuracy, categoricalAccuracy, get} from './metrics';
import {describeMathCPUAndGPU, expectTensorsClose} from './utils/test_utils';

describeMathCPUAndGPU('binaryAccuracy', () => {
  it('1D exact', () => {
    const x = tensor1d([1, 1, 1, 1, 0, 0, 0, 0]);
    const y = tensor1d([1, 0, 1, 0, 0, 1, 0, 1]);
    const accuracy = binaryAccuracy(x, y);
    expectTensorsClose(accuracy, scalar(0.5));
  });
  it('2D thresholded', () => {
    const x = tensor1d([1, 1, 1, 1, 0, 0, 0, 0]);
    const y = tensor1d([0.2, 0.4, 0.6, 0.8, 0.2, 0.3, 0.4, 0.7]);
    const accuracy = binaryAccuracy(x, y);
    expectTensorsClose(accuracy, scalar(5 / 8));
  });
  it('2D exact', () => {
    const x = tensor2d([[1, 1, 1, 1], [0, 0, 0, 0]], [2, 4]);
    const y = tensor2d([[1, 0, 1, 0], [0, 0, 0, 1]], [2, 4]);
    const accuracy = binaryAccuracy(x, y);
    expectTensorsClose(accuracy, tensor1d([0.5, 0.75]));
  });
  it('2D thresholded', () => {
    const x = tensor2d([[1, 1], [1, 1], [0, 0], [0, 0]], [4, 2]);
    const y =
        tensor2d([[0.2, 0.4], [0.6, 0.8], [0.2, 0.3], [0.4, 0.7]], [4, 2]);
    const accuracy = binaryAccuracy(x, y);
    expectTensorsClose(accuracy, tensor1d([0, 1, 1, 0.5]));
  });
});

describeMathCPUAndGPU('categoricalAccuracy', () => {
  it('1D', () => {
    const x = tensor1d([0, 0, 0, 1]);
    const y = tensor1d([0.1, 0.8, 0.05, 0.05]);
    const accuracy = categoricalAccuracy(x, y);
    expect(accuracy.dtype).toEqual('float32');
    expect(accuracy.shape).toEqual([]);
    expect(Array.from(accuracy.dataSync())).toEqual([0]);
  });
  it('2D', () => {
    const x = tensor2d([[0, 0, 0, 1], [0, 0, 0, 1]], [2, 4]);
    const y =
        tensor2d([[0.1, 0.8, 0.05, 0.05], [0.1, 0.05, 0.05, 0.8]], [2, 4]);
    const accuracy = categoricalAccuracy(x, y);
    expect(accuracy.dtype).toEqual('float32');
    expect(accuracy.shape).toEqual([2]);
    expect(Array.from(accuracy.dataSync())).toEqual([0, 1]);
  });
});

describe('metrics.get', () => {
  it('valid name, not alias', () => {
    expect(get('binaryAccuracy') === get('categoricalAccuracy')).toEqual(false);
  });
  it('valid name, alias', () => {
    expect(get('mse') === get('MSE')).toEqual(true);
  });
  it('invalid name', () => {
    expect(() => get('InvalidMetricName')).toThrowError(/Unknown metric/);
  });
  it('LossOrMetricFn input', () => {
    expect(get(binaryAccuracy)).toEqual(binaryAccuracy);
    expect(get(categoricalAccuracy)).toEqual(categoricalAccuracy);
  });
});
