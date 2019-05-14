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

import * as detectBrowser from 'detect-browser';
import * as math from 'mathjs';

import * as tfconverter from '@tensorflow/tfjs-converter';
import * as tfc from '@tensorflow/tfjs-core';
import * as tfd from '@tensorflow/tfjs-data';
import * as tfl from '@tensorflow/tfjs-layers';

// tslint:disable-next-line:max-line-length
import {BrowserEnvironmentType, BrowserEnvironmentInfo, ModelBenchmarkRun, ModelFunctionName, ModelTrainingBenchmarkRun, VersionSet, PythonEnvironmentInfo, NodeEnvironmentType, NodeEnvironmentInfo, EnvironmentInfo} from '../types';
// tslint:disable-next-line:max-line-length
import {addEnvironmentInfoToFirestore, addOrGetTaskId, addBenchmarkRunsToFirestore, addVersionSetToFirestore} from '../firestore';
import {NamedTensorMap} from '@tensorflow/tfjs-converter/dist/src/data/types';

// tslint:disable-next-line:no-any
let tfn: any;

export type MultiFunctionModelTaskLog = {[taskName: string]: ModelBenchmarkRun};

export interface SuiteLog {
  data: {[modelName: string]: MultiFunctionModelTaskLog};
  environmentInfo: PythonEnvironmentInfo;
  versionSet: VersionSet;
}

// tslint:disable-next-line:no-any
declare let __karma__: any;

/** Determine whether this file is running in Node.js. */
export function inNodeJS(): boolean {
  // Note this is not a generic way of testing if we are in Node.js.
  // The logic here is specific to the scripts in this folder.
  return typeof module !== 'undefined' && typeof process !== 'undefined' &&
      typeof __karma__ === 'undefined';
}

export function usingNodeGPU(): boolean {
  return process.argv.indexOf('--gpu') !== -1;
}

/** Extract commit hashes of the tfjs repos from karma flags. */
function getCommitHashesFromArgs(
    args: Array<boolean|number|string>) {
  for (let i = 0; i < args.length; ++i) {
    if (args[i] === '--hashes') {
      if (args[i + 1] == null) {
        throw new Error('Missing value for flag --hashes');
      }
      return JSON.parse(args[i + 1] as string);
    }
  }
}

/** Extract the "log" boolean flag from karma flags. */
function getLogFlagFromKarmaFlags(
    karmaFlags: Array<boolean|number|string>): boolean {
  for (let i = 0; i < karmaFlags.length; ++i) {
    if (karmaFlags[i] === '--log') {
      return karmaFlags[i + 1] === true;
    }
  }
  return false;
}

/**
 * Get model names in the order in which they are benchmarked in Python.
 *
 * @param suiteLog
 * @returns Model names sorted by ascending timestamps.
 */
function getChronologicalModelNames(suiteLog: SuiteLog): string[] {
  const modelNamesAndTimestamps: Array<{
    modelName: string,
    timestamp: number
  }> = [];
  for (const modelName in suiteLog.data) {
    const taskGroupLog = suiteLog.data[modelName] as MultiFunctionModelTaskLog;
    const functionNames = Object.keys(taskGroupLog);
    if (functionNames.length === 0) {
      continue;
    }

    const timestamp =
        new Date(taskGroupLog[functionNames[0]].endingTimestampMs).getTime();
    modelNamesAndTimestamps.push({modelName, timestamp});
  }

  modelNamesAndTimestamps.sort((x, y) => x.timestamp - y.timestamp);
  return modelNamesAndTimestamps.map(object => object.modelName);
}

/**
 * Make inputs and outputs of random values based on model's topology.
 *
 * @param model The model in question. Models with multiple inputs and/or
 *   outputs are supported.
 * @param batchSize Batch size: number of examples.
 * @returns An object with keys:
 *   - xs: concrete input tensors with uniform-random values.
 *   - ys: concrete output tensors with uniform-random values.
 */
function getRandomInputsAndOutputs(
    model: tfconverter.GraphModel | tfl.LayersModel, batchSize: number):
    {xs: tfc.Tensor|tfc.Tensor[], ys: tfc.Tensor|tfc.Tensor[]} {
  return tfc.tidy(() => {
    let xs: tfc.Tensor|tfc.Tensor[] = [];
    for (const input of model.inputs) {
      xs.push(tfc.randomUniform([batchSize].concat(input.shape.slice(1))));
    }
    if (xs.length === 1) {
      xs = xs[0];
    }

    let ys: tfc.Tensor|tfc.Tensor[] = [];
    if (model instanceof tfl.LayersModel) {
      for (const output of model.outputs) {
        ys.push(tfc.randomUniform([batchSize].concat(output.shape.slice(1))));
      }
      if (ys.length === 1) {
        ys = ys[0];
      }
    }

    return {xs, ys};
  });
}

/** Call await data() on tensor(s). */
async function syncData(tensors: tfc.Tensor|tfc.Tensor[]|NamedTensorMap) {
  if (tensors instanceof tfc.Tensor) {
    await (tensors as tfc.Tensor).data();
  } else  if (Array.isArray(tensors)) {
    const promises = tensors.map(tensor => tensor.data());
    await Promise.all(promises);
  } else {
    // tensors is a NamedTensorMap.
    const promises = Object.entries(tensors).map(item => item[1].data());
    await Promise.all(promises);
  }
}

/**
 * Get browser environment type.
 *
 * The type information includes the browser name and operating-system name.
 */
function getBrowserEnvironmentType(): BrowserEnvironmentType {
  const osName = detectBrowser.detectOS(navigator.userAgent).toLowerCase();
  const browserName = detectBrowser.detect().name.toLowerCase();
  return `${browserName}-${osName}` as BrowserEnvironmentType;
}

/**
 * Get browser environment information.
 *
 * The info includes: browser type and userAgent string.
 */
function getBrowserEnvironmentInfo(): BrowserEnvironmentInfo {
  return {
    type: getBrowserEnvironmentType(),
    userAgent: navigator.userAgent,
    // TODO(cais): Add WebGL info.
  };
}

function getNodeEnvironmentType(): NodeEnvironmentType {
  // TODO(cais): Support 'node-gles'.
  return usingNodeGPU() ? 'node-libtensorflow-cuda' : 'node-libtensorflow-cpu';
}

function getNodeEnvironmentInfo(): NodeEnvironmentInfo {
  const tfjsNodeVersion = tfn == null ? undefined : tfn.version['tfjs-node'];
  return {
    type: getNodeEnvironmentType(),
    nodeVersion: process.version,
    tfjsNodeVersion,
    tfjsNodeUsesCUDA: usingNodeGPU()
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
  const DATA_SERVER_ROOT = './base/data';
  const BENCHMARKS_JSON_URL = `${DATA_SERVER_ROOT}/benchmarks.json`;
  const BENCHMARKS_JSON_PATH = './data/benchmarks.json';

  /** Helper method for loading tf.LayersModel. */
  async function loadLayersModel(modelName: string): Promise<tfl.LayersModel> {
    // tslint:disable-next-line:no-any
    let modelJSON: any;
    if (inNodeJS()) {
      // In Node.js.
      const modelJSONPath = `./data/${modelName}/model.json`;
      // tslint:disable-next-line:no-require-imports
      const fs = require('fs');
      modelJSON = JSON.parse(fs.readFileSync(modelJSONPath, 'utf-8'));
    } else {
      // In browser.
      const modelJSONPath = `${DATA_SERVER_ROOT}/${modelName}/model.json`;
      modelJSON = await (await fetch(modelJSONPath)).json();
    }
    return await tfl.models.modelFromJSON(modelJSON['modelTopology']);
  }

  /** Helper method for loading tf.GraphModel. */
  async function loadGraphModel(modelName: string):
      Promise<tfconverter.GraphModel> {
    if (inNodeJS()) {
      // tslint:disable-next-line:no-require-imports
      const fileSystem = require('@tensorflow/tfjs-node/dist/io/file_system');
      return await tfconverter.loadGraphModel(
          fileSystem.fileSystem(`./data/${modelName}/model.json`));
    } else {
      return await tfconverter.loadGraphModel(
          `${DATA_SERVER_ROOT}/${modelName}/model.json`);
    }
  }

  it('Benchmark models', async () => {
    const isNodeJS = inNodeJS();
    console.log(`isNodeJS = ${isNodeJS}`);
    if (isNodeJS) {
      if (usingNodeGPU()) {
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
      log = getLogFlagFromKarmaFlags(__karma__.config.args);
    }
    console.log(`Boolean flag log = ${log}`);

    const taskType = 'model';
    let environmentInfo: EnvironmentInfo;
    if (isNodeJS) {
      environmentInfo = getNodeEnvironmentInfo();
    } else {
      environmentInfo = getBrowserEnvironmentInfo();
    }
    const versionSet: VersionSet = isNodeJS ? {
      versions: tfn.version
    } : {
      versions: {
        'tfjs-converter': tfconverter.version_converter,
        'tfjs-core': tfc.version_core,
        'tfjs-data': tfd.version_data,
        'tfjs-layers': tfl.version_layers
      }
    };

    let suiteLog: SuiteLog;
    if (isNodeJS) {
      // tslint:disable-next-line:no-require-imports
      const fs = require('fs');
      suiteLog = JSON.parse(fs.readFileSync(BENCHMARKS_JSON_PATH, 'utf-8'));
    } else {
      suiteLog = await (await fetch(BENCHMARKS_JSON_URL)).json() as SuiteLog;
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
      versionSet.commitHashes = getCommitHashesFromArgs(process.argv);
    } else {
      versionSet.commitHashes = getCommitHashesFromArgs(__karma__.config.args);
    }

    if (log) {
      console.log(
          `environmentId = ${tfjsEnvironmentId}; ` +
          `versionId = ${versionSetId}; ` +
          `pyEnvironmentId: ${pyEnvironmentId}`);
    }

    const sortedModelNames = getChronologicalModelNames(suiteLog);

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

      let model: tfconverter.GraphModel | tfl.LayersModel;
      if (modelFormat === 'LayersModel') {
        model = await loadLayersModel(modelName);
      } else if (modelFormat === 'GraphModel') {
        model = await loadGraphModel(modelName);
      } else {
        throw new Error(
            `Unsupported modelFormat: ${JSON.stringify(modelFormat)}`);
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
            await addOrGetTaskId(taskType, modelName, functionName) : null;

        checkBatchSize(pyRun.batchSize);

        let tfjsRun: ModelBenchmarkRun;

        const {xs, ys} = getRandomInputsAndOutputs(model, pyRun.batchSize);
        if (functionName === 'predict') {
          // Warm-up predict() runs.
          for (let n = 0; n < pyRun.numWarmUpIterations; ++n) {
            const predictOut = model.predict(xs);
            await syncData(predictOut);
            tfc.dispose(predictOut);
          }

          // Benchmarked predict() runs.
          const ts: number[] = [];
          for (let n = 0; n < pyRun.numBenchmarkedIterations; ++n) {
            const t0 = tfc.util.now();
            const predictOut = model.predict(xs);
            await syncData(predictOut);
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
      console.log(
          `Writing ${pyRuns.length} Python TaskLogs to Firestore...`);
      await addBenchmarkRunsToFirestore(pyRuns);
      console.log(`Done.`);
    }
  });
});
