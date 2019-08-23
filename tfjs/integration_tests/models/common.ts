
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
// tslint:disable-next-line: no-imports-from-dist
import {NamedTensorMap} from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';
import * as detectBrowser from 'detect-browser';

// tslint:disable-next-line:max-line-length
import {BrowserEnvironmentInfo, BrowserEnvironmentType, ModelBenchmarkRun, NodeEnvironmentInfo, NodeEnvironmentType, PythonEnvironmentInfo, ValidationRun, VersionSet} from '../types';

// tslint:disable-next-line:no-any
declare let __karma__: any;

// Karma data server root directory.
export const DATA_SERVER_ROOT = './base/data';

export type MultiFunctionModelTaskLog = {
  [taskName: string]: ModelBenchmarkRun|ValidationRun
};

export interface SuiteLog {
  data: {[modelName: string]: MultiFunctionModelTaskLog};
  environmentInfo: PythonEnvironmentInfo;
  versionSet: VersionSet;
}

/** Determine whether this file is running in Node.js. */
// tslint:disable-next-line:no-any
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
export function getCommitHashesFromArgs(args?: string[]) {
  if (args == null) {
    args = __karma__.config.args;
  }
  for (let i = 0; i < args.length; ++i) {
    if (args[i] === '--hashes') {
      if (args[i + 1] == null) {
        throw new Error('Missing value for flag --hashes');
      }
      return JSON.parse(args[i + 1]);
    }
  }
}

/** Extract the "log" boolean flag from karma flags. */
export function getLogFlagFromKarmaFlags(args?: Array<string|boolean>):
    boolean {
  if (args == null) {
    if (inNodeJS()) {
      throw new Error(
          'While in Node.js, you must pass command-line flags explicitly.');
    }
    args = __karma__.config.args;
  }
  for (let i = 0; i < args.length; ++i) {
    if (args[i] === '--log') {
      return args[i + 1] === true;
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
export function getChronologicalModelNames(suiteLog: SuiteLog): string[] {
  const modelNamesAndTimestamps: Array<{modelName: string, timestamp: number}> =
      [];
  for (const modelName in suiteLog.data) {
    const taskGroupLog = suiteLog.data[modelName];
    const functionNames = Object.keys(taskGroupLog);
    if (functionNames.length === 0) {
      continue;
    }

    const taskRun = taskGroupLog[functionNames[0]] as ModelBenchmarkRun;
    const timestamp = new Date(taskRun.endingTimestampMs).getTime();
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
export function getRandomInputsAndOutputs(
    model: tfconverter.GraphModel|tfl.LayersModel, batchSize: number):
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
export async function syncData(tensors: tfc.Tensor|tfc.Tensor[]|
                               NamedTensorMap) {
  if (tensors instanceof tfc.Tensor) {
    await tensors.data();
  } else if (Array.isArray(tensors)) {
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
export function getBrowserEnvironmentType(): BrowserEnvironmentType {
  const osName = detectBrowser.detectOS(navigator.userAgent).toLowerCase();
  const browserName = detectBrowser.detect().name.toLowerCase();
  return `${browserName}-${osName}` as BrowserEnvironmentType;
}

/**
 * Get browser environment information.
 *
 * The info includes: browser type and userAgent string.
 */
export function getBrowserEnvironmentInfo(): BrowserEnvironmentInfo {
  return {
    type: getBrowserEnvironmentType(),
    userAgent: navigator.userAgent,
    // TODO(cais): Add WebGL info.
  };
}

export function getNodeEnvironmentType(): NodeEnvironmentType {
  // TODO(cais): Support 'node-gles'.
  return usingNodeGPU() ? 'node-libtensorflow-cuda' : 'node-libtensorflow-cpu';
}

// tslint:disable-next-line:no-any
export function getNodeEnvironmentInfo(tfn: any): NodeEnvironmentInfo {
  const tfjsNodeVersion = tfn == null ? undefined : tfn.version['tfjs-node'];
  return {
    type: getNodeEnvironmentType(),
    nodeVersion: process.version,
    tfjsNodeVersion,
    tfjsNodeUsesCUDA: usingNodeGPU()
  };
}

export function checkBatchSize(batchSize: number) {
  if (!(Number.isInteger(batchSize) && batchSize > 0)) {
    throw new Error(
        `Expected batch size to be a positive integer, but got ` +
        `${batchSize}`);
  }
}

/** Helper method for loading tf.LayersModel. */
export async function loadLayersModel(modelName: string):
    Promise<tfl.LayersModel> {
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
  return tfl.models.modelFromJSON(modelJSON['modelTopology']);
}

/** Helper method for loading tf.GraphModel. */
export async function loadGraphModel(modelName: string):
    Promise<tfconverter.GraphModel> {
  if (inNodeJS()) {
    // tslint:disable-next-line:no-require-imports
    const fileSystem = require('@tensorflow/tfjs-node/dist/io/file_system');
    return tfconverter.loadGraphModel(
        fileSystem.fileSystem(`./data/${modelName}/model.json`));
  } else {
    return tfconverter.loadGraphModel(
        `${DATA_SERVER_ROOT}/${modelName}/model.json`);
  }
}
