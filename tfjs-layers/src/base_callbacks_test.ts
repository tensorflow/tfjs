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
 * Unit tests for callbacks.
 */

import {scalar, zeros} from '@tensorflow/tfjs-core';

import {BaseCallback, BaseLogger, CallbackConstructorRegistry, CallbackList, History} from './base_callbacks';
import {Callback} from './callbacks';
import {LayersModel} from './engine/training';
import * as tfl from './index';
import {disposeTensorsInLogs, Logs, resolveScalarsInLogs, UnresolvedLogs} from './logs';
import {describeMathCPUAndGPU} from './utils/test_utils';

describe('BaseLogger Callback', () => {
  it('Records and averages losses in an epoch', async () => {
    const baseLogger = new BaseLogger();
    baseLogger.setParams({metrics: ['loss', 'val_loss']});
    await baseLogger.onTrainBegin();
    await baseLogger.onEpochBegin(0);
    await baseLogger.onBatchBegin(0);
    await baseLogger.onBatchEnd(0, {batch: 0, size: 10, loss: 5});
    await baseLogger.onBatchBegin(1);
    await baseLogger.onBatchEnd(1, {batch: 1, size: 10, loss: 6});
    await baseLogger.onBatchBegin(2);
    await baseLogger.onBatchEnd(2, {batch: 2, size: 5, loss: 7});
    const epochLog: UnresolvedLogs = {val_loss: 3};
    await baseLogger.onEpochEnd(0, epochLog);
    expect(epochLog['val_loss'] as number).toEqual(3);
    expect(epochLog['loss'] as number)
        .toBeCloseTo((10 * 5 + 10 * 6 + 5 * 7) / (10 + 10 + 5));
  });
  it('Forgets old epochs', async () => {
    const baseLogger = new BaseLogger();
    baseLogger.setParams({metrics: ['loss', 'val_loss']});
    const numOldEpochs = 2;
    await baseLogger.onTrainBegin();
    for (let i = 0; i < numOldEpochs; ++i) {
      await baseLogger.onEpochBegin(i);
      await baseLogger.onBatchBegin(0);
      await baseLogger.onBatchEnd(0, {batch: 0, size: 10, loss: -5});
      const epochLog: UnresolvedLogs = {val_loss: 3};
      await baseLogger.onEpochEnd(i, epochLog);
    }
    await baseLogger.onEpochBegin(numOldEpochs);
    await baseLogger.onBatchBegin(0);
    await baseLogger.onBatchEnd(0, {batch: 0, size: 10, loss: 5});
    await baseLogger.onBatchBegin(1);
    await baseLogger.onBatchEnd(1, {batch: 1, size: 10, loss: 6});
    await baseLogger.onBatchBegin(2);
    await baseLogger.onBatchEnd(2, {batch: 2, size: 5, loss: 7});
    const epochLog: UnresolvedLogs = {val_loss: 3};
    await baseLogger.onEpochEnd(numOldEpochs, epochLog);
    expect(epochLog['val_loss'] as number).toEqual(3);
    expect(epochLog['loss'] as number)
        .toBeCloseTo((10 * 5 + 10 * 6 + 5 * 7) / (10 + 10 + 5));
  });
});

describe('CallbackList', () => {
  it('Constructor with empty arg', async () => {
    const callbackList = new CallbackList();
    await callbackList.onTrainBegin();
    await callbackList.onTrainEnd();
  });
  it('Constructor and setParams with array of callbacks', () => {
    const history1 = new History();
    const history2 = new History();
    const callbackList = new CallbackList([history1, history2]);
    const params = {'verbose': 3};
    callbackList.setParams(params);
    expect(history1.params).toEqual(params);
    expect(history2.params).toEqual(params);
  });
  it('onTrainBegin', async () => {
    const history1 = new History();
    const history2 = new History();
    const callbackList = new CallbackList([history1, history2]);
    await callbackList.onTrainBegin();
    expect(history1.epoch).toEqual([]);
    expect(history1.history).toEqual({});
    expect(history2.epoch).toEqual([]);
    expect(history2.history).toEqual({});
  });
  it('onEpochEnd', async () => {
    const history1 = new History();
    const history2 = new History();
    const callbackList = new CallbackList([history1, history2]);
    await callbackList.onTrainBegin();
    await callbackList.onEpochEnd(100, {'val_loss': 10, 'val_accuracy': 0.1});
    expect(history1.epoch).toEqual([100]);
    expect(history1.history).toEqual({'val_loss': [10], 'val_accuracy': [0.1]});
    expect(history2.epoch).toEqual([100]);
    expect(history2.history).toEqual({'val_loss': [10], 'val_accuracy': [0.1]});
    await callbackList.onEpochEnd(101, {'val_loss': 9.5, 'val_accuracy': 0.2});
    expect(history1.epoch).toEqual([100, 101]);
    expect(history1.history)
        .toEqual({'val_loss': [10, 9.5], 'val_accuracy': [0.1, 0.2]});
    expect(history2.epoch).toEqual([100, 101]);
    expect(history2.history)
        .toEqual({'val_loss': [10, 9.5], 'val_accuracy': [0.1, 0.2]});
  });
  it('append', async () => {
    const history1 = new History();
    const history2 = new History();
    const callbackList = new CallbackList([history1]);
    await callbackList.onTrainBegin();
    expect(history1.epoch).toEqual([]);
    expect(history1.history).toEqual({});
    await callbackList.append(history2);
    await callbackList.onTrainBegin();
    expect(history2.epoch).toEqual([]);
    expect(history2.history).toEqual({});
  });
});

describeMathCPUAndGPU('resolveScalarsInLogs', () => {
  it('Resolve mixed numbers and scalars', async () => {
    const logs: UnresolvedLogs = {
      'a': 1,
      'b': scalar(2),
      'c': -3,
      'd': scalar(-4),
    };
    await resolveScalarsInLogs(logs);
    expect(logs['a']).toEqual(1);
    expect(logs['b']).toEqual(2);
    expect(logs['c']).toEqual(-3);
    expect(logs['d']).toEqual(-4);
  });

  it('Resolve null works fine', async () => {
    const logs: UnresolvedLogs = null;
    await resolveScalarsInLogs(logs);
    expect(logs).toEqual(null);
  });

  it('Resolve empty works fine', async () => {
    const logs: UnresolvedLogs = {};
    await resolveScalarsInLogs(logs);
    expect(logs).toEqual({});
  });
});

describeMathCPUAndGPU('disposeTensorsInLogs', () => {
  it('Resolve mixed numbers and scalars', () => {
    const logs: UnresolvedLogs = {
      'a': 1,
      'b': scalar(2),
      'c': -3,
      'd': scalar(-4),
    };
    disposeTensorsInLogs(logs);
    expect(logs['a']).toEqual(1);
    // tslint:disable-next-line:no-any
    expect((logs['b'] as any).isDisposed).toEqual(true);
    expect(logs['c']).toEqual(-3);
    // tslint:disable-next-line:no-any
    expect((logs['d'] as any).isDisposed).toEqual(true);
  });
});

describe('History Callback', () => {
  it('onTrainBegin', async () => {
    const history = new History();
    await history.onTrainBegin();
    expect(history.epoch).toEqual([]);
    expect(history.history).toEqual({});
  });
  it('onEpochEnd', async () => {
    const history = new History();
    await history.onTrainBegin();
    await history.onEpochEnd(0, {'val_loss': 10, 'val_accuracy': 0.1});
    expect(history.epoch).toEqual([0]);
    expect(history.history).toEqual({'val_loss': [10], 'val_accuracy': [0.1]});
    await history.onEpochEnd(1, {'val_loss': 9.5, 'val_accuracy': 0.2});
    expect(history.epoch).toEqual([0, 1]);
    expect(history.history)
        .toEqual({'val_loss': [10, 9.5], 'val_accuracy': [0.1, 0.2]});
  });
});

class MockLayersModel extends LayersModel {
  constructor(name: string) {
    super({inputs: [], outputs: [], name});
  }
}

class MockCallback extends Callback {}

describe('CallbackList', () => {
  it('Constructor and setModel with array of callbacks', () => {
    const mockCallback1 = new MockCallback();
    const mockCallback2 = new MockCallback();
    const callbackList = new CallbackList([mockCallback1, mockCallback2]);
    const model = new MockLayersModel('MockModelA');
    callbackList.setModel(model);
    expect(mockCallback1.model).toEqual(model);
    expect(mockCallback2.model).toEqual(model);
  });
});

let fake1Epochs: number[];
class FakeCallback1 extends BaseCallback {
  constructor() {
    super();
    fake1Epochs = [];
  }
  async onEpochEnd(epoch: number, logs: Logs) {
    fake1Epochs.push(epoch);
  }
}

let fake2Epochs: number[];
class FakeCallback2 extends BaseCallback {
  constructor() {
    super();
    fake2Epochs = [];
  }
  async onEpochEnd(epoch: number, logs: Logs) {
    fake2Epochs.push(epoch);
  }
}

describe('CallbackConstructorRegistry', () => {
  beforeEach(() => {
    // tslint:disable-next-line:no-any
    (CallbackConstructorRegistry as any).clear();
  });

  it('Empty registry creates empty list of callbacks', () => {
    expect(CallbackConstructorRegistry.createCallbacks(0)).toEqual([]);
    expect(CallbackConstructorRegistry.createCallbacks(1)).toEqual([]);
    expect(CallbackConstructorRegistry.createCallbacks(2)).toEqual([]);
  });

  it('Registry with one element', () => {
    tfl.registerCallbackConstructor(1, FakeCallback1);

    let callbacks = CallbackConstructorRegistry.createCallbacks(0);
    expect(callbacks.length).toEqual(0);

    callbacks = CallbackConstructorRegistry.createCallbacks(1);
    expect(callbacks.length).toEqual(1);
    expect(callbacks[0] instanceof FakeCallback1).toEqual(true);

    callbacks = CallbackConstructorRegistry.createCallbacks(2);
    expect(callbacks.length).toEqual(1);
    expect(callbacks[0] instanceof FakeCallback1).toEqual(true);
  });

  it('Registry with two elements on two levels', () => {
    tfl.registerCallbackConstructor(1, FakeCallback1);
    tfl.registerCallbackConstructor(2, FakeCallback2);

    let callbacks = CallbackConstructorRegistry.createCallbacks(0);
    expect(callbacks.length).toEqual(0);

    callbacks = CallbackConstructorRegistry.createCallbacks(1);
    expect(callbacks.length).toEqual(1);
    expect(callbacks[0] instanceof FakeCallback1).toEqual(true);

    callbacks = CallbackConstructorRegistry.createCallbacks(2);
    expect(callbacks.length).toEqual(2);
    expect(callbacks[0] instanceof FakeCallback1).toEqual(true);
    expect(callbacks[1] instanceof FakeCallback2).toEqual(true);
  });

  it('Registry with two elements on the same level', () => {
    tfl.registerCallbackConstructor(2, FakeCallback1);
    tfl.registerCallbackConstructor(2, FakeCallback2);

    let callbacks = CallbackConstructorRegistry.createCallbacks(0);
    expect(callbacks.length).toEqual(0);

    callbacks = CallbackConstructorRegistry.createCallbacks(1);
    expect(callbacks.length).toEqual(0);

    callbacks = CallbackConstructorRegistry.createCallbacks(2);
    expect(callbacks.length).toEqual(2);
    expect(callbacks[0] instanceof FakeCallback1).toEqual(true);
    expect(callbacks[1] instanceof FakeCallback2).toEqual(true);
  });

  it('Duplicate registration on same level leads to Error', () => {
    tfl.registerCallbackConstructor(1, FakeCallback1);
    expect(() => tfl.registerCallbackConstructor(1, FakeCallback1))
        .toThrowError(/Duplicate callback constructor/);
  });

  it('Duplicate registration on different level leads to Error', () => {
    tfl.registerCallbackConstructor(1, FakeCallback1);
    expect(() => tfl.registerCallbackConstructor(2, FakeCallback1))
        .toThrowError(/Duplicate callback constructor/);
  });

  it('Invalid verbosityLevel leads to Error', () => {
    expect(() => tfl.registerCallbackConstructor(-1, FakeCallback1))
        .toThrowError(/is expected to be an integer >= 0/);
    expect(() => tfl.registerCallbackConstructor(0.5, FakeCallback1))
        .toThrowError(/is expected to be an integer >= 0/);
    expect(() => tfl.registerCallbackConstructor(NaN, FakeCallback1))
        .toThrowError(/is expected to be an integer >= 0/);
  });
});

describeMathCPUAndGPU('CallbackConstructorRegistry initialization', () => {
  beforeEach(() => {
    // tslint:disable-next-line:no-any
    (CallbackConstructorRegistry as any).clear();
  });
  it('CallbackConstructorRegistry is initialized properly', () => {
    expect(CallbackConstructorRegistry.createCallbacks(1)).toEqual([]);
  });
});

describeMathCPUAndGPU('LayersModel.fit and CallbackConstructorRegistry', () => {
  beforeEach(() => {
    // tslint:disable-next-line:no-any
    (CallbackConstructorRegistry as any).clear();
    fake1Epochs = [];
    fake2Epochs = [];
  });

  it('LayersModel.fit call with no callback ctor registered', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 1, inputShape: [1]}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const xs = zeros([4, 1]);
    const ys = zeros([4, 1]);
    await model.fit(xs, ys, {epochs: 3});

    expect(fake1Epochs).toEqual([]);
  });

  it('LayersModel.fit call with one callback ctor registered', async () => {
    tfl.registerCallbackConstructor(1, FakeCallback1);

    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 1, inputShape: [1]}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const xs = zeros([4, 1]);
    const ys = zeros([4, 1]);
    await model.fit(xs, ys, {epochs: 3});

    expect(fake1Epochs).toEqual([0, 1, 2]);
  });

  it('LayersModel.fit call with two callback ctor registered', async () => {
    tfl.registerCallbackConstructor(1, FakeCallback1);
    tfl.registerCallbackConstructor(2, FakeCallback2);

    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 1, inputShape: [1]}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const xs = zeros([4, 1]);
    const ys = zeros([4, 1]);

    await model.fit(xs, ys, {epochs: 3, verbose: 2});
    expect(fake1Epochs).toEqual([0, 1, 2]);
    expect(fake2Epochs).toEqual([0, 1, 2]);

    fake1Epochs = [];
    fake2Epochs = [];

    await model.fit(xs, ys, {epochs: 3, verbose: 1});
    expect(fake1Epochs).toEqual([0, 1, 2]);
    expect(fake2Epochs).toEqual([]);
  });
});
