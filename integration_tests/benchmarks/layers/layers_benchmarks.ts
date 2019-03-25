/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as math from 'mathjs';
import * as tf from '@tensorflow/tfjs';
import * as detectBrowser from 'detect-browser';

import {BenchmarkEnvironmentType, BrowserEnvironmentInfo, ModelTaskLog, MultiEnvironmentTaskLog, SuiteLog, TaskGroupLog} from '../types';

function getRandomInputsAndOutputs(model: tf.Model, batchSize: number):
    {xs: tf.Tensor|tf.Tensor[], ys: tf.Tensor|tf.Tensor[]} {
  return tf.tidy(() => {
    let xs: tf.Tensor|tf.Tensor[] = [];
    for (const input of model.inputs) {
      xs.push(tf.randomUniform([batchSize].concat(input.shape.slice(1))));
    }
    if (xs.length === 1) {
      xs = xs[0];
    }

    let ys: tf.Tensor|tf.Tensor[] = [];
    for (const output of model.outputs) {
      ys.push(tf.randomUniform([batchSize].concat(output.shape.slice(1))));
    }
    if (ys.length === 1) {
      ys = ys[0];
    }

    return {xs, ys};
  });
}

async function syncDataAndDispose(tensors: tf.Tensor|tf.Tensor[]) {
  if (Array.isArray(tensors)) {
    for (const out of tensors) {
      await out.data();
    }
  } else {
    await tensors.data();
  }
  tf.dispose(tensors);
}

function getBrowserEnvironmentType(): BenchmarkEnvironmentType {
  const osName = detectBrowser.detectOS(navigator.userAgent).toLowerCase();
  const browserName = detectBrowser.detect().name.toLowerCase();

  return `${browserName}-${osName}` as BenchmarkEnvironmentType;
}

describe('TF.js Layers Benchmarks', () => {
  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 600000;
  });

  // Karma serves static files under the base/ path.
  const DATA_SERVER_ROOT = './base/data';
  const BENCHMARKS_JSON_URL = `${DATA_SERVER_ROOT}/benchmarks.json`;

  async function getBenchmarkModelNamesAndConfig(): Promise<{
    modelNames: string[],
    predictNumBurnInRuns: number,
    predictNumBenchmarkRuns: number
  }> {
    const benchmarks = await (await fetch(BENCHMARKS_JSON_URL)).json();
    return {
      modelNames: benchmarks.models as string[],
      predictNumBurnInRuns: benchmarks.config.PREDICT_BURNINS as number,
      predictNumBenchmarkRuns: benchmarks.config.PREDICT_RUNS as number
    };
  }

  async function loadModel(modelName: string): Promise<tf.Model> {
    const modelJSONPath = `${DATA_SERVER_ROOT}/${modelName}/model.json`;
    const modelJSON = await (await fetch(modelJSONPath)).json();
    return await tf.models.modelFromJSON(modelJSON['modelTopology']);
  }

  async function loadModelBenchmarkData(modelName: string): Promise<any> {
    const modelJSONPath = `${DATA_SERVER_ROOT}/${modelName}/data.json`;
    return await (await fetch(modelJSONPath)).json();
  }

  it('Benchmark models', async () => {
    const {modelNames, predictNumBenchmarkRuns, predictNumBurnInRuns} =
        await getBenchmarkModelNamesAndConfig();

    const environmentType = getBrowserEnvironmentType();
    const environment: BrowserEnvironmentInfo = {
      type: environmentType,
      userAgent: navigator.userAgent
    };

    const suiteLog: SuiteLog = {
      data: {},
      metadata: {
        commitHashes: {
          'tfjs': 'foo',  // TODO(cais): Do not hard-code.
          'tfjs-core': 'foo',
          'tfjs-converter': 'foo',
          'tfjs-layers': 'foo',
          'tfjs-data': 'foo'
        },
        timestamp: new Date().toISOString()
      }
    };

    for (let i = 0; i < modelNames.length; ++i) {
      const modelName = modelNames[i];
      console.log(
          `Benchmark task ${i + 1} of ${modelNames.length}: ${modelName}`);
      const model = await loadModel(modelName);
      const benchmarkData = await loadModelBenchmarkData(modelName);

      const batchSize = benchmarkData.batch_size as number;

      const {xs, ys} = getRandomInputsAndOutputs(model, batchSize);

      // Burn-in runs.
      let predictOut: tf.Tensor|tf.Tensor[];
      for (let i = 0; i < predictNumBurnInRuns; ++i) {
        predictOut = model.predict(xs);
        await syncDataAndDispose(predictOut);
      }

      // Benchmarked runs.
      const ts: number[] = [];
      for (let i = 0; i < predictNumBenchmarkRuns; ++i) {
        const t0 = performance.now();
        predictOut = model.predict(xs);
        await syncDataAndDispose(predictOut);
        ts.push(performance.now() - t0);
      }

      const modelTaskLog: ModelTaskLog = {
        modelName,
        taskName: 'predict',
        numWarmUpRuns: predictNumBurnInRuns,
        numBenchmarkedRuns: predictNumBenchmarkRuns,
        batchSize,
        averageTimeMs: math.mean(ts),
        medianTimeMs: math.median(ts),
        minTimeMs: math.min(ts),
        timestamp: new Date().getTime(),
        environment
      };
      const multiEnvironmentTaskLog: MultiEnvironmentTaskLog = {};
      multiEnvironmentTaskLog[environmentType] = modelTaskLog;
      const taskGroupLog: TaskGroupLog = {
        'predict': multiEnvironmentTaskLog
      };
      suiteLog.data[modelName] = taskGroupLog;
  
      console.log(modelTaskLog);

      tf.dispose({xs, ys});
    }
    console.log(JSON.stringify(suiteLog, null, 2));
  });
});
