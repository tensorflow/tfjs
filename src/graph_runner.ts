/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {Tensor} from './graph';
import {InputProvider} from './input_provider';
import {NDArrayMath} from './math/math';
import {NDArray, Scalar} from './math/ndarray';
import {Optimizer} from './optimizer';
import {CostReduction, FeedEntry, Session} from './session';

const DEFAULT_EVAL_INTERVAL_MS = 1500;
const DEFAULT_COST_INTERVAL_MS = 500;
const DEFAULT_INFERENCE_EXAMPLE_INTERVAL_MS = 3000;

export interface GraphRunnerEventObserver {
  batchesTrainedCallback?: (totalBatchesTrained: number) => void;
  avgCostCallback?: (avgCost: Scalar) => void;
  metricCallback?: (metric: NDArray) => void;
  inferenceExamplesCallback?:
      (feeds: FeedEntry[][], inferenceValues: NDArray[]) => void;
  inferenceExamplesPerSecCallback?: (examplesPerSec: number) => void;
  trainExamplesPerSecCallback?: (examplesPerSec: number) => void;
  totalTimeCallback?: (totalTimeSec: number) => void;
  doneTrainingCallback?: () => void;
}

export enum MetricReduction {
  SUM,
  MEAN
}

/**
 * A class that drives the training of a graph model given a dataset. It allows
 * the user to provide a set of callbacks for measurements like cost, accuracy,
 * and speed of training.
 */
export class GraphRunner {
  private costTensor: Tensor;
  private trainFeedEntries: FeedEntry[];
  private batchSize: number;
  private optimizer: Optimizer;
  private currentTrainLoopNumBatches: number|undefined;
  private costIntervalMs: number;

  private metricTensor: Tensor|undefined;
  private metricFeedEntries: FeedEntry[]|undefined;
  private metricBatchSize: number|undefined;
  private metricReduction: MetricReduction;
  private metricIntervalMs: number;

  private inferenceTensor: Tensor;
  private inferenceFeedEntries: FeedEntry[]|undefined;
  private inferenceExampleIntervalMs: number;
  private inferenceExampleCount: number;

  // Runtime information.
  private isTraining: boolean;
  private totalBatchesTrained: number;
  private batchesTrainedThisRun: number;
  private lastComputedMetric: NDArray;

  private isInferring: boolean;
  private currentInferenceLoopNumPasses: number|undefined;
  private inferencePassesThisRun: number;

  private trainStartTimestamp: number;
  private lastCostTimestamp = 0;
  private lastEvalTimestamp = 0;

  private lastStopTimestamp: number|null;
  private totalIdleTimeMs = 0;

  private zeroScalar: Scalar;
  private metricBatchSizeScalar: Scalar;

  constructor(
      private math: NDArrayMath, private session: Session,
      private eventObserver: GraphRunnerEventObserver) {
    this.resetStatistics();
    this.zeroScalar = Scalar.new(0);
  }

  resetStatistics() {
    this.totalBatchesTrained = 0;
    this.totalIdleTimeMs = 0;
    this.lastStopTimestamp = null;
  }

  /**
   * Start the training loop with an optional number of batches to train for.
   * Optionally takes a metric tensor and feed entries to compute periodically.
   * This can be used for computing accuracy, or a similar metric.
   */
  train(
      costTensor: Tensor, trainFeedEntries: FeedEntry[], batchSize: number,
      optimizer: Optimizer, numBatches?: number, metricTensor?: Tensor,
      metricFeedEntries?: FeedEntry[], metricBatchSize?: number,
      metricReduction = MetricReduction.MEAN,
      evalIntervalMs = DEFAULT_EVAL_INTERVAL_MS,
      costIntervalMs = DEFAULT_COST_INTERVAL_MS) {
    this.costTensor = costTensor;
    this.trainFeedEntries = trainFeedEntries;
    this.metricTensor = metricTensor;
    this.metricFeedEntries = metricFeedEntries;
    if (metricBatchSize != null && this.metricBatchSize !== metricBatchSize) {
      if (this.metricBatchSizeScalar != null) {
        this.metricBatchSizeScalar.dispose();
      }
      this.metricBatchSizeScalar = Scalar.new(metricBatchSize);
    }
    this.metricBatchSize = metricBatchSize;
    this.metricReduction = metricReduction;
    this.batchSize = batchSize;
    this.optimizer = optimizer;

    this.metricIntervalMs = evalIntervalMs;
    this.costIntervalMs = costIntervalMs;
    this.currentTrainLoopNumBatches = numBatches;

    this.batchesTrainedThisRun = 0;
    this.isTraining = true;
    this.trainStartTimestamp = performance.now();
    this.trainNetwork();
  }

  stopTraining() {
    this.isTraining = false;
    this.lastStopTimestamp = performance.now();
  }

  resumeTraining() {
    this.isTraining = true;
    if (this.lastStopTimestamp != null) {
      this.totalIdleTimeMs += performance.now() - this.lastStopTimestamp;
    }
    this.trainNetwork();
  }

  private trainNetwork() {
    if (this.batchesTrainedThisRun === this.currentTrainLoopNumBatches) {
      this.stopTraining();
    }

    if (!this.isTraining) {
      if (this.eventObserver.doneTrainingCallback != null) {
        this.eventObserver.doneTrainingCallback();
      }
      return;
    }

    const start = performance.now();
    const shouldComputeCost = this.eventObserver.avgCostCallback != null &&
        (start - this.lastCostTimestamp > this.costIntervalMs);
    if (shouldComputeCost) {
      this.lastCostTimestamp = start;
    }

    const costReduction =
        shouldComputeCost ? CostReduction.MEAN : CostReduction.NONE;

    this.math.scope((keep) => {
      const avgCost = this.session.train(
          this.costTensor, this.trainFeedEntries, this.batchSize,
          this.optimizer, costReduction);

      if (shouldComputeCost) {
        const trainTime = performance.now() - start;

        this.eventObserver.avgCostCallback!(avgCost);

        if (this.eventObserver.trainExamplesPerSecCallback != null) {
          const examplesPerSec = (this.batchSize * 1000 / trainTime);
          this.eventObserver.trainExamplesPerSecCallback(examplesPerSec);
        }
      }

      if (this.eventObserver.metricCallback != null &&
          this.metricFeedEntries != null &&
          start - this.lastEvalTimestamp > this.metricIntervalMs) {
        this.lastEvalTimestamp = start;

        if (this.lastComputedMetric != null) {
          this.lastComputedMetric.dispose();
        }
        this.lastComputedMetric = this.computeMetric();
        this.eventObserver.metricCallback(this.lastComputedMetric);
      }

      if (this.eventObserver.totalTimeCallback != null) {
        this.eventObserver.totalTimeCallback(
            (start - this.trainStartTimestamp) / 1000);
      }

      this.batchesTrainedThisRun++;
      this.totalBatchesTrained++;

      if (this.eventObserver.batchesTrainedCallback != null) {
        this.eventObserver.batchesTrainedCallback(this.totalBatchesTrained);
      }

    });
    requestAnimationFrame(() => this.trainNetwork());
  }

  infer(
      inferenceTensor: Tensor, inferenceFeedEntries: FeedEntry[],
      inferenceExampleIntervalMs = DEFAULT_INFERENCE_EXAMPLE_INTERVAL_MS,
      inferenceExampleCount = 5, numPasses?: number) {
    if (this.eventObserver.inferenceExamplesCallback == null &&
        this.eventObserver.inferenceExamplesPerSecCallback == null) {
      throw new Error(
          'Cannot start inference loop, no inference example or ' +
          'examples/sec observer provided.');
    }

    // Make sure the feed values are providers, and not NDArrays.
    for (let i = 0; i < inferenceFeedEntries.length; i++) {
      const feedEntry = inferenceFeedEntries[i];

      if (feedEntry.data instanceof NDArray) {
        throw new Error(
            'Cannot start inference on the model runner with feed entries of ' +
            'type NDArray. Please use InputProviders.');
      }
    }

    this.inferenceExampleIntervalMs = inferenceExampleIntervalMs;
    this.inferenceTensor = inferenceTensor;
    this.inferenceFeedEntries = inferenceFeedEntries;
    this.inferenceExampleCount = inferenceExampleCount;
    this.currentInferenceLoopNumPasses = numPasses;
    if (!this.isInferring) {
      this.inferencePassesThisRun = 0;
      requestAnimationFrame(() => this.inferNetwork());
    }
    this.isInferring = true;
  }

  private inferNetwork() {
    if (!this.isInferring ||
        this.inferencePassesThisRun === this.currentInferenceLoopNumPasses) {
      return;
    }

    this.math.scope((keep, track) => {
      const feeds: FeedEntry[][] = [];
      const inferenceValues: NDArray[] = [];

      const start = performance.now();
      for (let i = 0; i < this.inferenceExampleCount; i++) {
        // Populate a new FeedEntry[] populated with NDArrays.
        const ndarrayFeedEntries: FeedEntry[] = [];
        for (let j = 0; j < this.inferenceFeedEntries!.length; j++) {
          const feedEntry = this.inferenceFeedEntries![j];
          ndarrayFeedEntries.push({
            tensor: feedEntry.tensor,
            data:
                track((feedEntry.data as InputProvider).getNextCopy(this.math))
          });
        }
        feeds.push(ndarrayFeedEntries);

        inferenceValues.push(
            this.session.eval(this.inferenceTensor, ndarrayFeedEntries));
      }

      if (this.eventObserver.inferenceExamplesPerSecCallback != null) {
        // Force a GPU download, since inference results are generally needed on
        // the CPU and it's more fair to include blocking on the GPU to complete
        // its work for the inference measurement.
        inferenceValues[inferenceValues.length - 1].getValues();

        const inferenceExamplesPerSecTime = performance.now() - start;

        const examplesPerSec =
            (this.inferenceExampleCount * 1000 / inferenceExamplesPerSecTime);
        this.eventObserver.inferenceExamplesPerSecCallback!(examplesPerSec);
      }

      if (this.eventObserver.inferenceExamplesCallback != null) {
        this.eventObserver.inferenceExamplesCallback(feeds, inferenceValues);
      }
      this.inferencePassesThisRun++;

    });
    setTimeout(() => this.inferNetwork(), this.inferenceExampleIntervalMs);
  }

  stopInferring() {
    this.isInferring = false;
  }

  isInferenceRunning(): boolean {
    return this.isInferring;
  }

  computeMetric(): Scalar {
    if (this.metricFeedEntries == null) {
      throw new Error('Cannot compute metric, no metric FeedEntries provided.');
    }

    let metric = this.zeroScalar;

    return this.math.scope((keep) => {
      for (let i = 0; i < this.metricBatchSize!; i++) {
        const metricValue =
            this.session.eval(this.metricTensor!, this.metricFeedEntries!);

        metric = this.math.add(metric, metricValue);
      }

      if (this.metricReduction === MetricReduction.MEAN) {
        metric = this.math.divide(metric, this.metricBatchSizeScalar);
      }

      return metric;
    });
  }

  getTotalBatchesTrained(): number {
    return this.totalBatchesTrained;
  }

  getLastComputedMetric(): Scalar {
    return this.lastComputedMetric;
  }

  setMath(math: NDArrayMath) {
    this.math = math;
  }

  setSession(session: Session) {
    this.session = session;
  }

  setInferenceTensor(inferenceTensor: Tensor) {
    this.inferenceTensor = inferenceTensor;
  }

  setInferenceExampleCount(inferenceExampleCount: number) {
    this.inferenceExampleCount = inferenceExampleCount;
  }
}
