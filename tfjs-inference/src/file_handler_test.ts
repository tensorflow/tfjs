/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';
import * as fs from 'fs';
import * as path from 'path';
import * as rimraf from 'rimraf';

import {FileHandler} from './file_handler';

/**
 * Create a temporary directory for writing files to.
 * @return {string} The absolute path of the tmp dir.
 */
function makeTmpDir() {
  return fs.mkdtempSync('tfjs_node_fs_test');
}

describe('File Handler', () => {
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

  let testDir: string;

  beforeEach(() => {
    testDir = makeTmpDir();
  });

  afterEach(() => {
    rimraf.sync(testDir);
  });

  describe('load json model', () => {
    it('load: two weight files', async done => {
      const weightsManifest: tf.io.WeightsManifestConfig = [
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
      const modelJSON: tf.io.ModelJSON = {
        modelTopology: modelTopology1,
        weightsManifest,
        signature: {},
        userDefinedMetadata: {},
        modelInitializer: {}
      };

      // Write model.json file.
      const modelJSONPath = path.join(testDir, 'model.json');
      fs.writeFileSync(modelJSONPath, JSON.stringify(modelJSON), 'utf8');

      // Write the two binary weights files.
      const weightsData1 =
          Buffer.from(new Float32Array([-1.1, -3.3, -3.3]).buffer);
      fs.writeFileSync(
          path.join(testDir, 'weights.1.bin'), weightsData1, 'binary');
      const weightsData2 = Buffer.from(new Float32Array([-7.7]).buffer);
      fs.writeFileSync(
          path.join(testDir, 'weights.2.bin'), weightsData2, 'binary');

      // Load the artifacts consisting of a model.json and two binary weight
      // files.
      const handler = new FileHandler(modelJSONPath);
      const modelArtifacts = await handler.load();

      expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
      expect(modelArtifacts.signature).toEqual({});
      expect(modelArtifacts.userDefinedMetadata).toEqual({});
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
      tf.test_util.expectArraysClose(
          new Float32Array(modelArtifacts.weightData),
          new Float32Array([-1.1, -3.3, -3.3, -7.7]));

      done();
    });
    it('loading from nonexistent model.json path fails', async done => {
      const modelJSONPath = path.join(testDir, 'foo', 'model.json');
      const handler = new FileHandler(modelJSONPath);

      try {
        await handler.load();
        done.fail('Loading from nonexistent file succeeded unexpectedly.');
      } catch (err) {
        expect(err.message).toMatch(/.*no such file or directory.*/);
      }

      done();
    });

    it('loading from missing weights path fails', async done => {
      const weightsManifest = [
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
      fs.writeFileSync(modelJSONPath, JSON.stringify(modelJSON), 'utf8');

      // Write only first of the two binary weights files.
      const weightsData1 =
          Buffer.from(new Float32Array([-1.1, -3.3, -3.3]).buffer);
      fs.writeFileSync(
          path.join(testDir, 'weights.1.bin'), weightsData1, 'binary');

      // Load the artifacts consisting of a model.json and two binary weight
      // files.
      try {
        const handler = new FileHandler(modelJSONPath);
        await handler.load();

        done.fail(
            'Loading with missing weights file succeeded ' +
            'unexpectedly.');
      } catch (err) {
        expect(err.message).toMatch(/.*no such file or directory.*/);
        done();
      }
    });
  });
});
