/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as tfc from '@tensorflow/tfjs-core';
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';
import * as fs from 'fs';
import * as path from 'path';
import * as rimraf from 'rimraf';
import {promisify} from 'util';

describe('File system IOHandler', () => {
  const mkdtemp = promisify(fs.mkdtemp);
  const readFile = promisify(fs.readFile);
  const writeFile = promisify(fs.writeFile);
  const rimrafPromise = promisify(rimraf);

  const modelTopology1: {} = {
    'class_name': 'Sequential',
    'keras_version': '2.1.6',
    'config': [{
      'class_name': 'Dense',
      'config': {
        'kernel_initializer': {
          'class_name': 'VarianceScaling',
          'config': {
            'distribution': 'uniform',
            'scale': 1.0,
            'seed': null,
            'mode': 'fan_avg'
          }
        },
        'name': 'dense',
        'kernel_constraint': null,
        'bias_regularizer': null,
        'bias_constraint': null,
        'dtype': 'float32',
        'activation': 'linear',
        'trainable': true,
        'kernel_regularizer': null,
        'bias_initializer': {'class_name': 'Zeros', 'config': {}},
        'units': 1,
        'batch_input_shape': [null, 3],
        'use_bias': true,
        'activity_regularizer': null
      }
    }],
    'backend': 'tensorflow'
  };
  const weightSpecs1: tfc.io.WeightsManifestEntry[] = [
    {
      name: 'dense/kernel',
      shape: [3, 1],
      dtype: 'float32',
    },
    {
      name: 'dense/bias',
      shape: [1],
      dtype: 'float32',
    }
  ];
  const weightData1 = new ArrayBuffer(16);

  let testDir: string;
  beforeEach(async done => {
    testDir = await mkdtemp('tfjs_node_fs_test');
    done();
  });

  afterEach(async done => {
    await rimrafPromise(testDir);
    done();
  });

  it('save succeeds with newly created directory', async done => {
    const t0 = new Date();
    testDir = path.join(testDir, 'save-destination');
    const handler = tfc.io.getSaveHandlers(`file://${testDir}`)[0];
    handler
        .save({
          modelTopology: modelTopology1,
          weightSpecs: weightSpecs1,
          weightData: weightData1,
        })
        .then(async saveResult => {
          expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
              .toBeGreaterThanOrEqual(t0.getTime());
          expect(saveResult.modelArtifactsInfo.modelTopologyType)
              .toEqual('JSON');

          const modelJSONPath = path.join(testDir, 'model.json');
          const weightsBinPath = path.join(testDir, 'weights.bin');
          const modelJSON = JSON.parse(await readFile(modelJSONPath, 'utf8'));
          expect(modelJSON.modelTopology).toEqual(modelTopology1);
          expect(modelJSON.weightsManifest.length).toEqual(1);
          expect(modelJSON.weightsManifest[0].paths).toEqual(['weights.bin']);
          expect(modelJSON.weightsManifest[0].weights).toEqual(weightSpecs1);

          const weightData = new Uint8Array(await readFile(weightsBinPath));
          expect(weightData.length).toEqual(16);
          weightData.forEach(value => expect(value).toEqual(0));

          // Verify the content of the files.
          done();
        })
        .catch(err => done.fail(err.stack));
  });

  it('save fails if path exists as a file', async done => {
    testDir = path.join(testDir, 'save-destination');
    // Create a file at the locatin.
    await writeFile(testDir, 'foo');
    const handler = tfc.io.getSaveHandlers(`file://${testDir}`)[0];
    handler
        .save({
          modelTopology: modelTopology1,
          weightSpecs: weightSpecs1,
          weightData: weightData1,
        })
        .then(saveResult => {
          done.fail('Saving to path of existing file succeeded unexpectedly.');
        })
        .catch(err => {
          expect(err.message).toMatch(/.*exists as a file.*directory.*/);
          done();
        });
  });

  it('save-load round trip: one weight file', done => {
    const handler1 = tfc.io.getSaveHandlers(`file://${testDir}`)[0];
    handler1
        .save({
          modelTopology: modelTopology1,
          weightSpecs: weightSpecs1,
          weightData: weightData1,
        })
        .then(saveResult => {
          const modelJSONPath = path.join(testDir, 'model.json');
          const handler2 = tfc.io.getLoadHandlers(`file://${modelJSONPath}`)[0];
          handler2.load()
              .then(modelArtifacts => {
                expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
                expect(modelArtifacts.weightSpecs).toEqual(weightSpecs1);
                expect(new Float32Array(modelArtifacts.weightData))
                    .toEqual(new Float32Array([0, 0, 0, 0]));
                done();
              })
              .catch(err => done.fail(err.stack));
        })
        .catch(err => done.fail(err.stack));
  });

  it('load: two weight files', async done => {
    const weightsManifest: tfc.io.WeightsManifestConfig = [
      {
        paths: ['weights.1.bin'],
        weights: [{
          name: 'dense/kernel',
          shape: [3, 1],
          dtype: 'float32',
        }],
      },

      {
        paths: ['weights.2.bin'],
        weights: [{
          name: 'dense/bias',
          shape: [1],
          dtype: 'float32',
        }]
      }
    ];
    const modelJSON = {
      modelTopology: modelTopology1,
      weightsManifest,
    };

    // Write model.json file.
    const modelJSONPath = path.join(testDir, 'model.json');
    await writeFile(modelJSONPath, JSON.stringify(modelJSON), 'utf8');

    // Write the two binary weights files.
    const weightsData1 =
        Buffer.from(new Float32Array([-1.1, -3.3, -3.3]).buffer);
    await writeFile(
        path.join(testDir, 'weights.1.bin'), weightsData1, 'binary');
    const weightsData2 = Buffer.from(new Float32Array([-7.7]).buffer);
    await writeFile(
        path.join(testDir, 'weights.2.bin'), weightsData2, 'binary');

    // Load the artifacts consisting of a model.json and two binary weight
    // files.
    const handler = tfc.io.getLoadHandlers(`file://${modelJSONPath}`)[0];
    handler.load()
        .then(modelArtifacts => {
          expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
          expect(modelArtifacts.weightSpecs).toEqual([
            {
              name: 'dense/kernel',
              shape: [3, 1],
              dtype: 'float32',
            },
            {
              name: 'dense/bias',
              shape: [1],
              dtype: 'float32',
            }
          ]);
          expectArraysClose(
              new Float32Array(modelArtifacts.weightData),
              new Float32Array([-1.1, -3.3, -3.3, -7.7]));
          done();
        })
        .catch(err => done.fail(err.stack));
  });

  it('loading from nonexistent model.json path fails', done => {
    const handler =
        tfc.io.getLoadHandlers(`file://${testDir}/foo/model.json`)[0];
    handler.load()
        .then(getModelArtifactsInfoForJSON => {
          done.fail(
              'Loading from nonexisting model.json path succeeded ' +
              'unexpectedly.');
        })
        .catch(err => {
          expect(err.message)
              .toMatch(/model\.json.*does not exist.*loading failed/);
          done();
        });
  });

  it('loading from missing weights path fails', async done => {
    const weightsManifest: tfc.io.WeightsManifestConfig = [
      {
        paths: ['weights.1.bin'],
        weights: [{
          name: 'dense/kernel',
          shape: [3, 1],
          dtype: 'float32',
        }],
      },

      {
        paths: ['weights.2.bin'],
        weights: [{
          name: 'dense/bias',
          shape: [1],
          dtype: 'float32',
        }]
      }
    ];
    const modelJSON = {
      modelTopology: modelTopology1,
      weightsManifest,
    };

    // Write model.json file.
    const modelJSONPath = path.join(testDir, 'model.json');
    await writeFile(modelJSONPath, JSON.stringify(modelJSON), 'utf8');

    // Write only first of the two binary weights files.
    const weightsData1 =
        Buffer.from(new Float32Array([-1.1, -3.3, -3.3]).buffer);
    await writeFile(
        path.join(testDir, 'weights.1.bin'), weightsData1, 'binary');

    // Load the artifacts consisting of a model.json and two binary weight
    // files.
    const handler = tfc.io.getLoadHandlers(`file://${modelJSONPath}`)[0];
    handler.load()
        .then(modelArtifacts => {
          done.fail(
              'Loading with missing weights file succeeded ' +
              'unexpectedly.');
        })
        .catch(err => {
          expect(err.message)
              .toMatch(/Weight file .*weights\.2\.bin does not exist/);
          done();
        });
  });
});
