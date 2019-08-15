/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {tensor2d} from '@tensorflow/tfjs-core';

import * as tfl from './index';
import {describeMathCPUAndGPU} from './utils/test_utils';

describe('EarlyStopping', () => {
  function createDummyModel(): tfl.LayersModel {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 1, inputShape: [1]}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    return model;
  }

  it('Default monitor, default mode, increasing val_loss', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping();
    callback.setModel(model);

    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 9});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(2);
    await callback.onEpochEnd(2, {val_loss: 9.5});
    expect(model.stopTraining).toEqual(true);
  });

  it('Default monitor, default mode, holding val_loss', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping();
    callback.setModel(model);

    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 9});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(2);
    await callback.onEpochEnd(2, {val_loss: 9});
    expect(model.stopTraining).toEqual(true);
  });

  it('Default monitor, default mode, custom minDelta', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping({minDelta: 1});
    callback.setModel(model);

    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 8});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(2);
    // An decrease of 0.5 is < minDelta (1) and should trigger stop.
    await callback.onEpochEnd(2, {val_loss: 7.5});
    expect(model.stopTraining).toEqual(true);
  });

  it('Default monitor, default mode, custom baseline, stopping', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping({baseline: 5});
    callback.setModel(model);

    await callback.onTrainBegin();
    await callback.onEpochBegin(1);
    // Failure to go below the baseline will stop the training immediately.
    await callback.onEpochEnd(1, {val_loss: 6});
    expect(model.stopTraining).toEqual(true);
  });

  it('Default monitor, default mode, custom baseline, not stopping',
     async () => {
       const model = createDummyModel();
       const callback = tfl.callbacks.earlyStopping({baseline: 5});
       callback.setModel(model);

       await callback.onTrainBegin();
       await callback.onEpochBegin(1);
       // If the loss value goes below the baseline, training should continue.
       await callback.onEpochEnd(1, {val_loss: 4});
       expect(model.stopTraining).toBeUndefined();
       // If the loss value increases, training should stop;
       await callback.onEpochEnd(1, {val_loss: 4.5});
       expect(model.stopTraining).toEqual(true);
     });

  it('Custom monitor, default model, increasing', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping({monitor: 'aux_loss'});
    callback.setModel(model);

    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10, aux_loss: 100});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 9, aux_loss: 120});
    expect(model.stopTraining).toEqual(true);
  });

  it('Custom monitor, max, increasing', async () => {
    const model = createDummyModel();
    const callback =
        tfl.callbacks.earlyStopping({monitor: 'aux_metric', mode: 'max'});
    callback.setModel(model);

    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10, aux_metric: 100});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 9, aux_metric: 120});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(2);
    await callback.onEpochEnd(2, {val_loss: 9, aux_metric: 110});
    expect(model.stopTraining).toEqual(true);
  });

  it('Custom monitor, max, custom minDelta', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping(
        {monitor: 'aux_metric', mode: 'max', minDelta: 5});
    callback.setModel(model);

    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10, aux_metric: 100});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 9, aux_metric: 120});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(2);
    // An increase of 2 is < minDelta (5) and should cause stopping.
    await callback.onEpochEnd(2, {val_loss: 9, aux_metric: 122});
    expect(model.stopTraining).toEqual(true);
  });

  it('Custom monitor, max, custom negative minDelta', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping(
        {monitor: 'aux_metric', mode: 'max', minDelta: -5});
    callback.setModel(model);

    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10, aux_metric: 100});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 9, aux_metric: 120});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(2);
    // An increase of 2 is < minDelta (5) and should cause stopping.
    await callback.onEpochEnd(2, {val_loss: 9, aux_metric: 122});
    expect(model.stopTraining).toEqual(true);
  });

  it('Patience = 2', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping({patience: 2});
    callback.setModel(model);

    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 9});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(2);
    await callback.onEpochEnd(2, {val_loss: 9.5});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(3);
    await callback.onEpochEnd(3, {val_loss: 9.6});
    expect(model.stopTraining).toEqual(true);
  });

  it('Missing monitor leads to warning', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping();
    callback.setModel(model);

    const warnMessages: string[] = [];
    function fakeWarn(message: string) {
      warnMessages.push(message);
    }
    spyOn(console, 'warn').and.callFake(fakeWarn);

    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    // Note that the default monitor (val_loss) is missing.
    await callback.onEpochEnd(0, {loss: 100});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {loss: 100});
    expect(model.stopTraining).toBeUndefined();

    expect(warnMessages.length).toEqual(2);
    expect(warnMessages[0]).toMatch(/val_loss is not available/);
    expect(warnMessages[1]).toMatch(/val_loss is not available/);
  });
});

describeMathCPUAndGPU('EarlyStopping LayersModel.fit() integration', () => {
  it('Functional model, monitor loss, With minDelta', async () => {
    const input = tfl.input({shape: [1]});
    const output =
        tfl.layers.dense({units: 1, kernelInitializer: 'ones'}).apply(input) as
        tfl.SymbolicTensor;
    const model = tfl.model({inputs: input, outputs: output});
    const xs = tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tensor2d([2, 3, 4, 5], [4, 1]);
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // Without the EarlyStopping callback, the loss value would be:
    //   1, 0.734, 0.549, 0.421, 0.332, ...
    // With loss being monitored and minDelta set to 0.25, the training should
    // stop after the 3rd epoch.
    const history = await model.fit(xs, ys, {
      epochs: 10,
      callbacks: tfl.callbacks.earlyStopping({monitor: 'loss', minDelta: 0.25})
    });
    expect(history.history.loss.length).toEqual(3);
  });

  it('Sequential model, monitor val_acc', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      activation: 'softmax',
      kernelInitializer: 'ones',
      inputShape: [2]
    }));
    const xs = tensor2d([1, 2, 3, 4], [2, 2]);
    const ys = tensor2d([[1, 0, 0], [0, 1, 0]], [2, 3]);
    const xsVal = tensor2d([4, 3, 2, 1], [2, 2]);
    const ysVal = tensor2d([[0, 0, 1], [0, 1, 0]], [2, 3]);
    model.compile(
        {loss: 'categoricalCrossentropy', optimizer: 'sgd', metrics: ['acc']});

    // Without the EarlyStopping callback, the val_acc value would be:
    //   0.5, 0.5, 0.5, 0.5, ...
    // With val_acc being monitored, training should stop after the 2nd epoch.
    const history = await model.fit(xs, ys, {
      epochs: 10,
      validationData: [xsVal, ysVal],
      callbacks: tfl.callbacks.earlyStopping({monitor: 'val_acc'})
    });
    expect(history.history.loss.length).toEqual(2);
  });

  it('Sequential model, monitor val_acc, custom patience', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      activation: 'softmax',
      kernelInitializer: 'ones',
      inputShape: [2]
    }));
    const xs = tensor2d([1, 2, 3, 4], [2, 2]);
    const ys = tensor2d([[1, 0, 0], [0, 1, 0]], [2, 3]);
    const xsVal = tensor2d([4, 3, 2, 1], [2, 2]);
    const ysVal = tensor2d([[0, 0, 1], [0, 1, 0]], [2, 3]);
    model.compile(
        {loss: 'categoricalCrossentropy', optimizer: 'sgd', metrics: ['acc']});

    // Without the EarlyStopping callback, the val_acc value would be:
    //   0.5, 0.5, 0.5, 0.5, ...
    // With val_acc being monitored and patience set to 4, training should stop
    // after the 5th epoch.
    const history = await model.fit(xs, ys, {
      epochs: 10,
      validationData: [xsVal, ysVal],
      callbacks: tfl.callbacks.earlyStopping({monitor: 'val_acc', patience: 4})
    });
    expect(history.history.loss.length).toEqual(5);
  });
});
