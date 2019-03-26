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

import {logSuiteLog} from '../firebase';
import {SuiteLog, BenchmarkEnvironmentType, ModelTask, ModelTaskLog, BrowserEnvironmentInfo, ModelFitTaskLog} from '../types';

function getChronologicalModelNames(suiteLog: SuiteLog): string[] {
  const modelNamesAndTimestamps: Array<{
    modelName: string,
    timestamp: number
  }> = [];
  for (const modelName in suiteLog.data) {
    const taskGroupLog = suiteLog.data[modelName];
    const taskNames = Object.keys(taskGroupLog);
    if (taskNames.length === 0) {
      continue;
    }
    const multiEnvironmentTaskLog = taskGroupLog[taskNames[0]];
    const environmentTypes = Object.keys(multiEnvironmentTaskLog) as BenchmarkEnvironmentType[];
    if (environmentTypes.length === 0) {
      continue;
    }
    const taskLog = multiEnvironmentTaskLog[environmentTypes[0]];

    modelNamesAndTimestamps.push({
      modelName,
      timestamp: new Date(taskLog.timestamp).getTime()
    });
  }

  modelNamesAndTimestamps.sort((x, y) => x.timestamp - y.timestamp);
  return modelNamesAndTimestamps.map(object => object.modelName);
}

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

function getBrowserEnvironmentInfo(): BrowserEnvironmentInfo {
  return {
    type: getBrowserEnvironmentType(),
    userAgent: navigator.userAgent,
    // TODO(cais): Add WebGL info.
  };
}

function checkBatchSize(batchSize: number) {
  if (!(Number.isInteger(batchSize) && batchSize > 0)) {
    throw new Error(
      `Expected batch size to be a positive integer, but got ` +
      `${batchSize}`);
  }
}

const OPTIMIZER_MAP: {[pyName: string]: string} = {
  'AdamOptimizer': 'adam',
  'RMSPropOptimizer': 'rmsprop',
  'GradientDescentOptimizer': 'sgd'
};

const LOSS_MAP: {[pyName: string]: string} = {
  'mean_squared_error': 'meanSquaredError',
  'categorical_crossentropy': 'categoricalCrossentropy'
};

describe('TF.js Layers Benchmarks', () => {
  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 600000;
  });

  // Karma serves static files under the base/ path.
  const DATA_SERVER_ROOT = './base/data';
  const BENCHMARKS_JSON_URL = `${DATA_SERVER_ROOT}/benchmarks.json`;

  async function loadModel(modelName: string): Promise<tf.Model> {
    const modelJSONPath = `${DATA_SERVER_ROOT}/${modelName}/model.json`;
    const modelJSON = await (await fetch(modelJSONPath)).json();
    return await tf.models.modelFromJSON(modelJSON['modelTopology']);
  }

  it('Benchmark models', async () => {
    const environmentType = getBrowserEnvironmentType();
    const environmentInfo = getBrowserEnvironmentInfo();

    const suiteLog =
        await (await fetch(BENCHMARKS_JSON_URL)).json() as SuiteLog;

    const sortedModelNames = getChronologicalModelNames(suiteLog);

    for (let i = 0; i < sortedModelNames.length; ++i) {
      const modelName = sortedModelNames[i];
      const taskGroupLog = suiteLog.data[modelName];
      console.log(`${i + 1}/${sortedModelNames.length}: ${modelName}`);
      const model = await loadModel(modelName);

      const taskNames = Object.keys(taskGroupLog) as ModelTask[];
      if (taskNames.length === 0) {
        throw new Error(`No task is found for model "${modelName}"`);
      }
      for (let j = 0; j < taskNames.length; ++j) {
        const taskName = taskNames[j];
        const multiEnvironTaskLog = taskGroupLog[taskName];
        const pyEnviron =
            Object.keys(multiEnvironTaskLog) as BenchmarkEnvironmentType[];
        if (pyEnviron.length !== 1) {
          throw new Error(
              `Found two python environment types: ${pyEnviron}.` +
              `Expected exactly 1`);
        }
        let pyLog = multiEnvironTaskLog[pyEnviron[0]] as ModelTaskLog;
        checkBatchSize(pyLog.batchSize);

        const {xs, ys} = getRandomInputsAndOutputs(model, pyLog.batchSize);
        const tsLog = Object.assign({}, pyLog);
        if (taskName === 'predict') {
          // Warm-up predict() runs.
          for (let n = 0; n < pyLog.numWarmUpRuns; ++n) {
            await syncDataAndDispose(model.predict(xs));
          }

          // Benchmarked predict() runs.
          const ts: number[] = [];
          for (let n = 0; n < pyLog.numBenchmarkedRuns; ++n) {
            const t0 = tf.util.now();
            await syncDataAndDispose(model.predict(xs));
            ts.push(tf.util.now() - t0);
          }

          // Format data for predict().
          tsLog.averageTimeMs = math.mean(ts);
          tsLog.medianTimeMs = math.median(ts);
          tsLog.minTimeMs = math.median(ts);
          console.log(
              `  predict(): averageTimeMs: ` +
              `py=${pyLog.averageTimeMs.toFixed(3)}, ` +
              `ts=${tsLog.averageTimeMs.toFixed(3)}`);
        } else if (taskName === 'fit') {
          const pyFitLog = pyLog as ModelFitTaskLog;
          model.compile({
            loss: LOSS_MAP[pyFitLog.loss],
            optimizer: OPTIMIZER_MAP[pyFitLog.optimizer]
          });

          // Warm-up fit() call.
          await model.fit(xs, ys, {
            epochs: pyLog.numWarmUpRuns,
            yieldEvery: 'never'
          });

          // Benchmarked fit() call.
          const t0 = tf.util.now();
          await model.fit(xs, ys, {
            epochs: pyLog.numBenchmarkedRuns,
            yieldEvery: 'never'
          });
          const t = tf.util.now() - t0;

          // Format data for fit().
          tsLog.averageTimeMs = t / pyLog.numBenchmarkedRuns;
          console.log(
            `  fit(): averageTimeMs: ` +
            `py=${pyLog.averageTimeMs.toFixed(3)}, ` +
            `ts=${tsLog.averageTimeMs.toFixed(3)}`);
        } else {
          console.warn(`Skipping task "${taskName}" of model "${modelName}"`);
        }
        if (tsLog != null) {
          tsLog.environment = environmentInfo;
          multiEnvironTaskLog[environmentType] = tsLog;
        }

        tf.dispose({xs, ys});
      }
    }

    console.log('Logging to firebase...');
    await logSuiteLog(suiteLog);
  });
});
