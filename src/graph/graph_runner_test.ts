/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {ENV} from '../environment';
import {NDArrayMath} from '../math/math';
import {Array1D, NDArray, Scalar} from '../math/ndarray';
import {Optimizer} from '../math/optimizers/optimizer';
import {SGDOptimizer} from '../math/optimizers/sgd_optimizer';

import {Graph, Tensor} from './graph';
// tslint:disable-next-line:max-line-length
import {GraphRunner, GraphRunnerEventObserver, MetricReduction} from './graph_runner';
import {CostReduction, FeedEntry, Session} from './session';

const FAKE_LEARNING_RATE = 1.0;
const FAKE_BATCH_SIZE = 10;

function fakeTrainBatch(
    math: NDArrayMath, feedEntries: FeedEntry[], batchSize: number,
    optimizer: Optimizer, costReduction: CostReduction) {
  return Scalar.new(.5);
}

describe('Model runner', () => {
  const math = ENV.math;
  let g: Graph;
  let session: Session;
  let optimizer: SGDOptimizer;
  let inputTensor: Tensor;
  let labelTensor: Tensor;
  let costTensor: Tensor;
  let predictionTensor: Tensor;
  let metricTensor: Tensor;

  let graphRunner: GraphRunner;

  let avgCostCallback: (avgCost: Scalar) => void;
  let metricCallback: (metric: Scalar) => void;
  let originalTimeout: number;

  const fakeUserEvents: GraphRunnerEventObserver = {
    batchesTrainedCallback: (totalBatchesTrained: number) => null,
    avgCostCallback: (avgCost: Scalar) => avgCostCallback(avgCost),
    metricCallback: (metric: Scalar) => metricCallback(metric),
    inferenceExamplesCallback:
        (feeds: FeedEntry[][], inferenceValues: NDArray[]) => null,
    trainExamplesPerSecCallback: (examplesPerSec: number) => null,
    totalTimeCallback: (totalTime: number) => null
  };

  beforeEach(() => {
    // Workaround to avoid jasmine callback timeout.
    originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 20000;
    g = new Graph();
    optimizer = new SGDOptimizer(FAKE_LEARNING_RATE);

    inputTensor = g.placeholder('input', [2]);

    predictionTensor = g.add(inputTensor, g.constant(Array1D.new([1, 1])));

    labelTensor = g.placeholder('label', [2]);

    costTensor = g.softmaxCrossEntropyCost(predictionTensor, labelTensor);

    metricTensor = g.argmaxEquals(predictionTensor, labelTensor);

    session = new Session(g, math);

    spyOn(session, 'train').and.callFake(fakeTrainBatch);
    let counter = 0;
    spyOn(session, 'eval').and.callFake((evalTensor: Tensor) => {
      if (evalTensor === predictionTensor) {
        return Array1D.new([1, 0]);
      } else if (evalTensor === metricTensor) {
        return Scalar.new(counter++ % 2);
      } else {
        throw new Error('Eval tensor not recognized');
      }
    });
    spyOn(fakeUserEvents, 'batchesTrainedCallback').and.callThrough();
    spyOn(fakeUserEvents, 'avgCostCallback').and.callThrough();
    spyOn(fakeUserEvents, 'metricCallback').and.callThrough();
    spyOn(fakeUserEvents, 'inferenceExamplesCallback').and.callThrough();
    spyOn(fakeUserEvents, 'trainExamplesPerSecCallback').and.callThrough();
    spyOn(fakeUserEvents, 'totalTimeCallback').and.callThrough();
  });

  afterEach(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout;
  });

  it('basic train usage, train 3 batches', (doneFn) => {
    const numBatches = 3;
    const trainFeedEntries: FeedEntry[] = [];
    const testExampleProvider: FeedEntry[] = [];

    // Mark this async test as done once the model runner calls our callback,
    // and fail if it doesn't.
    fakeUserEvents.doneTrainingCallback = () => {
      for (let i = 0; i < numBatches; i++) {
        // All batches should compute cost.
        const args = (session.train as jasmine.Spy).calls.argsFor(i);
        expect(args).toEqual([
          costTensor, trainFeedEntries, FAKE_BATCH_SIZE, optimizer,
          CostReduction.MEAN
        ]);
        (fakeUserEvents.avgCostCallback as jasmine.Spy).calls.argsFor(i);
        (fakeUserEvents.metricCallback as jasmine.Spy).calls.argsFor(i);
      }
      expect((fakeUserEvents.avgCostCallback as jasmine.Spy).calls.count())
          .toEqual(numBatches);
      expect((fakeUserEvents.metricCallback as jasmine.Spy).calls.count())
          .toEqual(numBatches);
      expect((session.train as jasmine.Spy).calls.count()).toEqual(numBatches);

      // 2 test examples are provided per batch.
      expect((session.eval as jasmine.Spy).calls.count())
          .toEqual(FAKE_BATCH_SIZE * numBatches);
      expect((fakeUserEvents.avgCostCallback as jasmine.Spy).calls.count())
          .toEqual(numBatches);
      expect((fakeUserEvents.metricCallback as jasmine.Spy).calls.count())
          .toEqual(numBatches);
      expect((fakeUserEvents.trainExamplesPerSecCallback as jasmine.Spy)
                 .calls.count())
          .toEqual(numBatches);
      expect((fakeUserEvents.totalTimeCallback as jasmine.Spy).calls.count())
          .toEqual(numBatches);
      expect(
          (fakeUserEvents.batchesTrainedCallback as jasmine.Spy).calls.count())
          .toEqual(numBatches);
      expect(graphRunner.getTotalBatchesTrained()).toEqual(numBatches);

      // Inference callbacks should not be called during training.
      expect((fakeUserEvents.inferenceExamplesCallback as jasmine.Spy)
                 .calls.count())
          .toEqual(0);

      doneFn();
    };

    avgCostCallback = (avgCost: Scalar) => {
      expect(avgCost.get()).toEqual(.5);
    };
    metricCallback = (metric: Scalar) => {
      expect(metric.get()).toEqual(.5);
    };

    graphRunner = new GraphRunner(math, session, fakeUserEvents);

    expect(graphRunner.getTotalBatchesTrained()).toEqual(0);

    graphRunner.train(
        costTensor, trainFeedEntries, FAKE_BATCH_SIZE, optimizer, numBatches,
        metricTensor, testExampleProvider, FAKE_BATCH_SIZE,
        MetricReduction.MEAN, 0, 0);
  });

  it('basic inference usage', (doneFn) => {
    const intervalMs = 0;
    const exampleCount = 2;
    const numPasses = 1;

    fakeUserEvents.inferenceExamplesCallback =
        (inputInferenceExamples: FeedEntry[][],
         inferenceOutputs: NDArray[]) => {
          expect(inputInferenceExamples.length).toEqual(exampleCount);
          expect(inferenceOutputs.length).toEqual(exampleCount);
          expect((session.eval as jasmine.Spy).calls.count())
              .toEqual(exampleCount * numPasses);

          // No training should have occured.
          expect(graphRunner.getTotalBatchesTrained()).toEqual(0);
          expect((fakeUserEvents.avgCostCallback as jasmine.Spy).calls.count())
              .toEqual(0);
          expect((fakeUserEvents.metricCallback as jasmine.Spy).calls.count())
              .toEqual(0);
          expect((fakeUserEvents.trainExamplesPerSecCallback as jasmine.Spy)
                     .calls.count())
              .toEqual(0);
          expect(
              (fakeUserEvents.totalTimeCallback as jasmine.Spy).calls.count())
              .toEqual(0);
          expect((fakeUserEvents.batchesTrainedCallback as jasmine.Spy)
                     .calls.count())
              .toEqual(0);
          expect(graphRunner.getTotalBatchesTrained()).toEqual(0);
          doneFn();
        };
    graphRunner = new GraphRunner(math, session, fakeUserEvents);

    const inferenceFeedEntries: FeedEntry[] = [];
    graphRunner.infer(
        predictionTensor, inferenceFeedEntries, intervalMs, exampleCount,
        numPasses);
  });
});
