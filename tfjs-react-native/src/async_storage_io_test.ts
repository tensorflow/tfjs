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

import './platform_react_native';
import * as tf from '@tensorflow/tfjs-core';
import {asyncStorageIO} from './async_storage_io';

describe('AsyncStorageIO', () => {
  // Test data.
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

  const weightSpecs1: tf.io.WeightsManifestEntry[] = [
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

  const artifacts1: tf.io.ModelArtifacts = {
    modelTopology: modelTopology1,
    weightSpecs: weightSpecs1,
    weightData: weightData1,
    format: 'layers-model',
    generatedBy: 'TensorFlow.js v0.0.0',
    convertedBy: null
  };

  const artifactsV0: tf.io.ModelArtifacts = {
    modelTopology: modelTopology1,
    weightSpecs: weightSpecs1,
    weightData: weightData1
  };

  it('constructs an IOHandler', async () => {
    const handler = asyncStorageIO('foo');
    expect(typeof handler.load).toBe('function');
    expect(typeof handler.save).toBe('function');
  });

  it('save returns a SaveResult', async () => {
    const handler = asyncStorageIO('FooModel');
    const result = await handler.save(artifacts1);
    expect(result.modelArtifactsInfo).toBeDefined();
    expect(result.modelArtifactsInfo).not.toBeNull();

    expect(result.modelArtifactsInfo.modelTopologyType).toBe('JSON');
    expect(result.modelArtifactsInfo.weightDataBytes).toBe(16);
  });

  it('Save-load round trip succeeds', async () => {
    const handler = asyncStorageIO('FooModel');
    await handler.save(artifacts1);

    const handler2 = asyncStorageIO('FooModel');
    const loaded = await handler2.load();

    expect(loaded.modelTopology).toEqual(modelTopology1);
    expect(loaded.weightSpecs).toEqual(weightSpecs1);
    expect(loaded.weightData).toEqual(weightData1);
    expect(loaded.format).toEqual('layers-model');
    expect(loaded.generatedBy).toEqual('TensorFlow.js v0.0.0');
    expect(loaded.convertedBy).toEqual(null);
  });

  it('Save-load round trip succeeds: v0 format', async () => {
    const handler1 = asyncStorageIO('FooModel');
    await handler1.save(artifactsV0);

    const handler2 = asyncStorageIO('FooModel');
    const loaded = await handler2.load();

    expect(loaded.modelTopology).toEqual(modelTopology1);
    expect(loaded.weightSpecs).toEqual(weightSpecs1);
    expect(loaded.weightData).toEqual(weightData1);
    expect(loaded.format).toEqual(undefined);
    expect(loaded.generatedBy).toEqual(undefined);
    expect(loaded.convertedBy).toEqual(undefined);
  });
});
