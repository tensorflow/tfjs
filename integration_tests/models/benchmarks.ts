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

import * as tfconverter from '@tensorflow/tfjs-converter';
import * as tfc from '@tensorflow/tfjs-core';
import * as tfd from '@tensorflow/tfjs-data';
import * as tfl from '@tensorflow/tfjs-layers';
import * as math from 'mathjs';

// tslint:disable-next-line:max-line-length
import {addBenchmarkRunsToFirestore, addEnvironmentInfoToFirestore, addOrGetTaskId, addVersionSetToFirestore} from '../firestore';
// tslint:disable-next-line:max-line-length
import {EnvironmentInfo, ModelBenchmarkRun, ModelFunctionName, ModelTrainingBenchmarkRun, VersionSet} from '../types';

import * as common from './common';

// tslint:disable-next-line:no-any
let tfn: any;

const OPTIMIZER_MAP: {[pyName: string]: string} = {
  'AdamOptimizer': 'adam',
  'RMSPropOptimizer': 'rmsprop',
  'GradientDescentOptimizer': 'sgd',
  'SGD': 'sgd'
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
  const BENCHMARKS_JSON_URL = `${common.DATA_SERVER_ROOT}/benchmarks.json`;
  const BENCHMARKS_JSON_PATH = './data/benchmarks.json';

  it('Benchmark models', async () => {
    const isNodeJS = common.inNodeJS();
    console.log(`isNodeJS = ${isNodeJS}`);
    if (isNodeJS) {
      if (common.usingNodeGPU()) {
        console.log('Using tfjs-node-gpu');
      } else {
        console.log('Using tfjs-node');
      }
      // tslint:disable-next-line:no-require-imports
      tfn = require('@tensorflow/tfjs-node');
      // NOTE: Even though it may appear that we are always importing the CPU
      // version of tfjs-node, we are really import the CPU/GPU version
      // depending on the setting, because the tfjs-node package here is built
      // from source and linked via `yalc`, instead of downloaded from NPM.
    }

    let log: boolean;  //  Whether the benchmark results are to be logged.
    if (isNodeJS) {
      // In Node.js.
      log = process.argv.indexOf('--log') !== -1;
    } else {
      // In browser.
      log = common.getLogFlagFromKarmaFlags();
    }
    console.log(`Boolean flag log = ${log}`);

    const taskType = 'model';
    let environmentInfo: EnvironmentInfo;
    if (isNodeJS) {
      environmentInfo = common.getNodeEnvironmentInfo(tfn);
    } else {
      environmentInfo = common.getBrowserEnvironmentInfo();
    }
    const versionSet: VersionSet = isNodeJS ? {versions: tfn.version} : {
      versions: {
        'tfjs-converter': tfconverter.version_converter,
        'tfjs-core': tfc.version_core,
        'tfjs-data': tfd.version_data,
        'tfjs-layers': tfl.version_layers
      }
    };

    let suiteLog: common.SuiteLog;
    if (isNodeJS) {
      // tslint:disable-next-line:no-require-imports
      const fs = require('fs');
      suiteLog = JSON.parse(fs.readFileSync(BENCHMARKS_JSON_PATH, 'utf-8'));
    } else {
      suiteLog =
          await (await fetch(BENCHMARKS_JSON_URL)).json() as common.SuiteLog;
    }
    const pyEnvironmentInfo = suiteLog.environmentInfo;
    const pyEnvironmentId =
        log ? await addEnvironmentInfoToFirestore(pyEnvironmentInfo) : null;
    environmentInfo.systemInfo = pyEnvironmentInfo.systemInfo;
    environmentInfo.cpuInfo = pyEnvironmentInfo.cpuInfo;
    environmentInfo.memInfo = pyEnvironmentInfo.memInfo;

    // Add environment info to firestore and retrieve the doc ID.
    const tfjsEnvironmentId =
        log ? await addEnvironmentInfoToFirestore(environmentInfo) : null;
    const versionSetId =
        log ? await addVersionSetToFirestore(versionSet) : null;
    if (isNodeJS) {
      versionSet.commitHashes = common.getCommitHashesFromArgs(process.argv);
    } else {
      versionSet.commitHashes = common.getCommitHashesFromArgs();
    }

    if (log) {
      console.log(
          `environmentId = ${tfjsEnvironmentId}; ` +
          `versionId = ${versionSetId}; ` +
          `pyEnvironmentId: ${pyEnvironmentId}`);
    }

    const sortedModelNames = common.getChronologicalModelNames(suiteLog);

    console.log('Environment:');
    console.log(JSON.stringify(environmentInfo, null, 2));
    console.log('Version set:');
    console.log(JSON.stringify(versionSet, null, 2));

    const pyRuns: ModelBenchmarkRun[] = [];
    const tfjsRuns: ModelBenchmarkRun[] = [];

    for (let i = 0; i < sortedModelNames.length; ++i) {
      const modelName = sortedModelNames[i];
      const taskGroupLog = suiteLog.data[modelName];
      if (Object.keys(taskGroupLog).length === 0) {
        throw new Error(`Found no task log in model ${modelName}`);
      }
      const modelFormat =
          taskGroupLog[Object.keys(taskGroupLog)[0]].modelFormat;

      console.log(`${i + 1}/${sortedModelNames.length}: ${modelName}`);

      if (modelFormat == null) {
        continue;
      }
      let model: tfconverter.GraphModel|tfl.LayersModel;
      if (modelFormat === 'LayersModel') {
        model = await common.loadLayersModel(modelName);
      } else if (modelFormat === 'GraphModel') {
        model = await common.loadGraphModel(modelName);
      } else {
        throw new Error(
            `Unsupported modelFormat for model "${modelName}": ` +
            `${JSON.stringify(modelFormat)}`);
      }

      const functionNames = Object.keys(taskGroupLog) as ModelFunctionName[];
      if (functionNames.length === 0) {
        throw new Error(`No task is found for model "${modelName}"`);
      }
      for (let j = 0; j < functionNames.length; ++j) {
        const functionName = functionNames[j];
        const pyRun = taskGroupLog[functionName] as ModelBenchmarkRun;

        // Make sure that the task type is logged in Firestore.
        const taskId = log ?
            await addOrGetTaskId(taskType, modelName, functionName) :
            null;

        common.checkBatchSize(pyRun.batchSize);

        let tfjsRun: ModelBenchmarkRun;

        const {xs, ys} =
            common.getRandomInputsAndOutputs(model, pyRun.batchSize);
        if (functionName === 'predict') {
          // Warm-up predict() runs.
          for (let n = 0; n < pyRun.numWarmUpIterations; ++n) {
            const predictOut = model.predict(xs);
            await common.syncData(predictOut);
            tfc.dispose(predictOut);
          }

          // Benchmarked predict() runs.
          const ts: number[] = [];
          for (let n = 0; n < pyRun.numBenchmarkedIterations; ++n) {
            const t0 = tfc.util.now();
            const predictOut = model.predict(xs);
            await common.syncData(predictOut);
            ts.push(tfc.util.now() - t0);
            tfc.dispose(predictOut);
          }

          // Format data for predict().
          tfjsRun = {
            taskId,
            taskType,
            modelFormat,
            modelName,
            // TODO(cais): Add modelId.
            functionName,
            batchSize: pyRun.batchSize,
            versionSetId,
            environmentId: tfjsEnvironmentId,
            numWarmUpIterations: pyRun.numWarmUpIterations,
            numBenchmarkedIterations: pyRun.numBenchmarkedIterations,
            timesMs: ts,
            averageTimeMs: math.mean(ts),
            endingTimestampMs: new Date().getTime()
          };
          console.log(
              `  (taskId=${taskId}) predict(): averageTimeMs: ` +
              `py=${pyRun.averageTimeMs.toFixed(3)}, ` +
              `tfjs=${tfjsRun.averageTimeMs.toFixed(3)}`);
        } else if (functionName === 'fit') {
          if (model instanceof tfconverter.GraphModel) {
            throw new Error('GraphModel does not support training');
          }
          const pyFitLog = pyRun as ModelTrainingBenchmarkRun;
          model.compile({
            loss: LOSS_MAP[pyFitLog.loss],
            optimizer: OPTIMIZER_MAP[pyFitLog.optimizer]
          });

          // Warm-up fit() call.
          await model.fit(xs, ys, {
            epochs: pyRun.numWarmUpIterations,
            yieldEvery: 'never',
            verbose: 0
          });

          // Benchmarked fit() call.
          const t0 = tfc.util.now();
          await model.fit(xs, ys, {
            epochs: pyRun.numBenchmarkedIterations,
            yieldEvery: 'never',
            verbose: 0
          });
          const t = tfc.util.now() - t0;

          // Format data for fit().
          tfjsRun = {
            taskId,
            taskType,
            modelFormat,
            modelName,
            // TODO(cais): Add modelId.
            functionName,
            batchSize: pyRun.batchSize,
            versionSetId,
            environmentId: tfjsEnvironmentId,
            numWarmUpIterations: pyRun.numWarmUpIterations,
            numBenchmarkedIterations: pyRun.numBenchmarkedIterations,
            averageTimeMs: t / pyRun.numBenchmarkedIterations,
            endingTimestampMs: new Date().getTime()
          };
          console.log(
              `  (taskId=${taskId}) fit(): averageTimeMs: ` +
              `py=${pyRun.averageTimeMs.toFixed(3)}, ` +
              `tfjs=${tfjsRun.averageTimeMs.toFixed(3)}`);
        } else {
          throw new Error(
              `Unknown task "${functionName}" from model "${modelName}"`);
        }

        tfc.dispose({xs, ys});

        tfjsRuns.push(tfjsRun);
        pyRun.taskId = taskId;
        pyRun.environmentId = pyEnvironmentId;
        pyRuns.push(pyRun);
      }
    }

    if (log) {
      console.log(
          `Writing ${tfjsRuns.length} TensorFlow.js TaskLogs to Firestore...`);
      await addBenchmarkRunsToFirestore(tfjsRuns);
      console.log(`Writing ${pyRuns.length} Python TaskLogs to Firestore...`);
      await addBenchmarkRunsToFirestore(pyRuns);
      console.log(`Done.`);
    }
  });
});
