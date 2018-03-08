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

import {BaseLogger, CallbackList, History, Logs} from './callbacks';
import {Model} from './engine/training';

class MockModel extends Model {
  constructor(name: string) {
    super({inputs: [], outputs: [], name});
  }
}

describe('BaseLogger Callback', () => {
  it('Records and averages losses in an epoch', async done => {
    const baseLogger = new BaseLogger();
    baseLogger.setParams({metrics: ['loss', 'val_loss']});
    await baseLogger.onEpochBegin(0);
    await baseLogger.onBatchBegin(0);
    await baseLogger.onBatchEnd(0, {batch: 0, size: 10, loss: 5});
    await baseLogger.onBatchBegin(1);
    await baseLogger.onBatchEnd(1, {batch: 1, size: 10, loss: 6});
    await baseLogger.onBatchBegin(2);
    await baseLogger.onBatchEnd(2, {batch: 2, size: 5, loss: 7});
    const epochLog: Logs = {val_loss: 3};
    await baseLogger.onEpochEnd(0, epochLog);
    expect(epochLog['val_loss'] as number).toEqual(3);
    expect(epochLog['loss'] as number)
        .toBeCloseTo((10 * 5 + 10 * 6 + 5 * 7) / (10 + 10 + 5));
    done();
  });
  it('Forgets old epochs', async done => {
    const baseLogger = new BaseLogger();
    baseLogger.setParams({metrics: ['loss', 'val_loss']});
    const numOldEpochs = 2;
    for (let i = 0; i < numOldEpochs; ++i) {
      await baseLogger.onEpochBegin(i);
      await baseLogger.onBatchBegin(0);
      await baseLogger.onBatchEnd(0, {batch: 0, size: 10, loss: -5});
      const epochLog: Logs = {val_loss: 3};
      await baseLogger.onEpochEnd(i, epochLog);
    }
    await baseLogger.onEpochBegin(numOldEpochs);
    await baseLogger.onBatchBegin(0);
    await baseLogger.onBatchEnd(0, {batch: 0, size: 10, loss: 5});
    await baseLogger.onBatchBegin(1);
    await baseLogger.onBatchEnd(1, {batch: 1, size: 10, loss: 6});
    await baseLogger.onBatchBegin(2);
    await baseLogger.onBatchEnd(2, {batch: 2, size: 5, loss: 7});
    const epochLog: Logs = {val_loss: 3};
    await baseLogger.onEpochEnd(numOldEpochs, epochLog);
    expect(epochLog['val_loss'] as number).toEqual(3);
    expect(epochLog['loss'] as number)
        .toBeCloseTo((10 * 5 + 10 * 6 + 5 * 7) / (10 + 10 + 5));
    done();
  });
});

describe('History Callback', () => {
  it('onTrainBegin', async done => {
    const history = new History();
    await history.onTrainBegin();
    expect(history.epoch).toEqual([]);
    expect(history.history).toEqual({});
    done();
  });
  it('onEpochEnd', async done => {
    const history = new History();
    await history.onTrainBegin();
    await history.onEpochEnd(0, {'val_loss': 10, 'val_accuracy': 0.1});
    expect(history.epoch).toEqual([0]);
    expect(history.history).toEqual({'val_loss': [10], 'val_accuracy': [0.1]});
    await history.onEpochEnd(1, {'val_loss': 9.5, 'val_accuracy': 0.2});
    expect(history.epoch).toEqual([0, 1]);
    expect(history.history)
        .toEqual({'val_loss': [10, 9.5], 'val_accuracy': [0.1, 0.2]});
    done();
  });
});

describe('CallbackList', () => {
  it('Constructor with empty arg', async done => {
    const callbackList = new CallbackList();
    await callbackList.onTrainBegin();
    await callbackList.onTrainEnd();
    done();
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
  it('Constructor and setModel with array of callbacks', () => {
    const history1 = new History();
    const history2 = new History();
    const callbackList = new CallbackList([history1, history2]);
    const model = new MockModel('MockModelA');
    callbackList.setModel(model);
    expect(history1.model).toEqual(model);
    expect(history2.model).toEqual(model);
  });
  it('onTrainBegin', async done => {
    const history1 = new History();
    const history2 = new History();
    const callbackList = new CallbackList([history1, history2]);
    await callbackList.onTrainBegin();
    expect(history1.epoch).toEqual([]);
    expect(history1.history).toEqual({});
    expect(history2.epoch).toEqual([]);
    expect(history2.history).toEqual({});
    done();
  });
  it('onEpochEnd', async done => {
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
    done();
  });
  it('append', async done => {
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
    done();
  });
});
