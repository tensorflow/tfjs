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

// tslint:disable-next-line:max-line-length
import {EnvironmentInfo, ModelFunctionName, ValidationRun, VersionSet} from '../types';

import * as common from './common';

// tslint:disable-next-line:no-any
let tfn: any;

// tslint:disable-next-line:no-any
declare let __karma__: any;

describe('TF.js converter validation', () => {
  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 600000;
  });

  // Karma serves static files under the base/ path.
  const VALIDATIONS_JSON_URL = `${common.DATA_SERVER_ROOT}/validations.json`;
  const VALIDATIONS_JSON_PATH = './data/validations.json';

  it('validate models', async () => {
    const isNodeJS = common.inNodeJS(__karma__);
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
      log = common.getLogFlagFromKarmaFlags(__karma__.config.args);
    }
    console.log(`Boolean flag log = ${log}`);

    let environmentInfo: EnvironmentInfo;
    if (isNodeJS) {
      environmentInfo = common.getNodeEnvironmentInfo(tfn);
    } else {
      environmentInfo = common.getBrowserEnvironmentInfo();
    }
    const versionSet: VersionSet = isNodeJS ? {versions: tfn.version} : {
      versions: {
        'tfjs-converter': tfconverter.version_converter,
        'tfjs-core': tfc.version_core
      }
    };

    let suiteLog: common.SuiteLog;
    if (isNodeJS) {
      // tslint:disable-next-line:no-require-imports
      const fs = require('fs');
      suiteLog = JSON.parse(fs.readFileSync(VALIDATIONS_JSON_PATH, 'utf-8'));
    } else {
      suiteLog =
          await (await fetch(VALIDATIONS_JSON_URL)).json() as common.SuiteLog;
    }
    const pyEnvironmentInfo = suiteLog.environmentInfo;
    environmentInfo.systemInfo = pyEnvironmentInfo.systemInfo;
    environmentInfo.cpuInfo = pyEnvironmentInfo.cpuInfo;
    environmentInfo.memInfo = pyEnvironmentInfo.memInfo;

    if (isNodeJS) {
      versionSet.commitHashes = common.getCommitHashesFromArgs(process.argv);
    } else {
      versionSet.commitHashes =
          common.getCommitHashesFromArgs(__karma__.config.args);
    }

    const sortedModelNames = common.getChronologicalModelNames(suiteLog);

    console.log('Environment:');
    console.log(JSON.stringify(environmentInfo, null, 2));
    console.log('Version set:');
    console.log(JSON.stringify(versionSet, null, 2));

    for (let i = 0; i < sortedModelNames.length; ++i) {
      const modelName = sortedModelNames[i];
      const taskGroupLog = suiteLog.data[modelName];
      console.log(Object.keys(taskGroupLog));
      if (Object.keys(taskGroupLog).length === 0) {
        throw new Error(`Found no task log in model ${modelName}`);
      }
      const modelFormat =
          taskGroupLog[Object.keys(taskGroupLog)[0]].modelFormat;

      console.log(`${i + 1}/${sortedModelNames.length}: ${modelName}`);

      let model: tfconverter.GraphModel;
      if (modelFormat === 'GraphModel') {
        model = await common.loadGraphModel(modelName);
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
        const task = taskGroupLog[functionName] as ValidationRun;
        const inputs =
            Object.entries(task.inputs).reduce((prev, [key, value]) => {
              prev[key] = tfc.tensor(value.value, value.shape, value.dtype);
              return prev;
            }, {} as tfc.NamedTensorMap);
        const outputs = Object.keys(task.outputs);
        if (functionName === 'predict') {
          if (task.async) {
            let predictOut =
                await model.executeAsync(inputs, outputs) as tfc.Tensor[];
            if (!Array.isArray(predictOut)) {
              predictOut = [predictOut];
            }
            outputs.map((key, index) => {
              expect(predictOut[index].dtype).toEqual(task.outputs[key].dtype);
              expect(predictOut[index].shape).toEqual(task.outputs[key].shape);
              tfc.test_util.expectArraysClose(
                  predictOut[index].dataSync(),
                  [].concat.apply([], task.outputs[key].value));
            });
            tfc.dispose(predictOut);
          } else {
            let predictOut = model.execute(inputs, outputs) as tfc.Tensor[];
            if (!Array.isArray(predictOut)) {
              predictOut = [predictOut];
            }
            outputs.map((key, index) => {
              expect(predictOut[index].dtype).toEqual(task.outputs[key].dtype);
              expect(predictOut[index].shape).toEqual(task.outputs[key].shape);
              tfc.test_util.expectArraysClose(
                  predictOut[index].dataSync(),
                  [].concat.apply([], task.outputs[key].value));
            });
            tfc.dispose(predictOut);
          }
        } else {
          throw new Error(
              `Unknown task "${functionName}" from model "${modelName}"`);
        }

        tfc.dispose(inputs);
      }
    }
  });
});
