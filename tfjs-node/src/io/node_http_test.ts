/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs';

import * as tfn from '../index';

// We still need node-fetch so that we can mock the core
// tf.env().platform.fetch call and return a valid response.
// tslint:disable-next-line:no-require-imports
const fetch = require('node-fetch');

const OCTET_STREAM_TYPE = 'application/octet-stream';
const JSON_TYPE = 'application/json';

// Test data;
const modelTopology1: {} = {
  'class_name': 'Sequential',
  'keras_version': '2.1.4',
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

describe('nodeHTTPRequest-load', () => {
  let requestInits: RequestInit[];
  const setupFakeWeightFiles = (fileBufferMap: {
    [filename: string]: string|Float32Array|Int32Array|ArrayBuffer|Uint8Array|
    Uint16Array
  }) => {
    spyOn(tf.env().platform, 'fetch')
        .and.callFake((path: string, init: RequestInit) => {
          return new Promise((resolve, reject) => {
            let contentType = '';
            if (path.endsWith('model.json')) {
              contentType = JSON_TYPE;
            } else if (
                path.endsWith('weightfile0') || path.endsWith('weightfile1')) {
              contentType = OCTET_STREAM_TYPE;
            } else {
              reject(new Error(`Invalid path: ${path}`));
            }
            requestInits.push(init);
            resolve(new fetch.Response(
                fileBufferMap[path],
                {'headers': {'Content-Type': contentType}}));
          });
        });
  };

  beforeEach(() => {
    requestInits = [];
  });

  it('Constructor', () => {
    const handler = tfn.io.nodeHTTPRequest('./foo_model.json');
    expect(handler == null).toEqual(false);
    expect(typeof handler.load).toEqual('function');
    expect(typeof handler.save).toEqual('function');
  });

  it('Load through NodeHTTPRequest object', async () => {
    const weightManifest1: tf.io.WeightsManifestConfig = [{
      paths: ['weightfile0'],
      weights: [
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
      ]
    }];
    const trainingConfig1: tf.io.TrainingConfig = {
      loss: 'categorical_crossentropy',
      metrics: ['accuracy'],
      optimizer_config: {class_name: 'SGD', config: {learningRate: 0.1}}
    };
    const floatData = new Float32Array([1, 3, 3, 7]);
    setupFakeWeightFiles({
      'http://localhost/model.json': JSON.stringify({
        modelTopology: modelTopology1,
        weightsManifest: weightManifest1,
        trainingConfig: trainingConfig1
      }),
      'http://localhost/weightfile0': floatData,
    });

    const handler = tfn.io.nodeHTTPRequest(
        'http://localhost/model.json',
        {credentials: 'include', cache: 'no-cache'});
    const modelArtifacts = await handler.load();
    expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
    expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
    expect(modelArtifacts.trainingConfig).toEqual(trainingConfig1);
    expect(new Float32Array(modelArtifacts.weightData)).toEqual(floatData);

    expect(requestInits).toEqual([
      {credentials: 'include', cache: 'no-cache'},
      {credentials: 'include', cache: 'no-cache'}
    ]);
  });

  it('Load through registered handler', async () => {
    const weightManifest1: tf.io.WeightsManifestConfig = [{
      paths: ['weightfile0'],
      weights: [
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
      ]
    }];
    const floatData = new Float32Array([1, 3, 3, 7]);
    setupFakeWeightFiles({
      'https://localhost/model.json': JSON.stringify(
          {modelTopology: modelTopology1, weightsManifest: weightManifest1}),
      'https://localhost/weightfile0': floatData,
    });

    const model = await tf.loadLayersModel('https://localhost/model.json');
    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, 3]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, 1]);
  });
});
