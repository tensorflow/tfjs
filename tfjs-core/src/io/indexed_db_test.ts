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

/**
 * Unit tests for indexed_db.ts.
 */

import * as tf from '../index';
import {BROWSER_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArrayBuffersEqual} from '../test_util';
import {browserIndexedDB, BrowserIndexedDB, BrowserIndexedDBManager, deleteDatabase, indexedDBRouter} from './indexed_db';

describeWithFlags('IndexedDB', BROWSER_ENVS, () => {
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
    convertedBy: null,
    modelInitializer: {}
  };

  const weightSpecs2: tf.io.WeightsManifestEntry[] = [
    {
      name: 'dense/new_kernel',
      shape: [5, 1],
      dtype: 'float32',
    },
    {
      name: 'dense/new_bias',
      shape: [1],
      dtype: 'float32',
    }
  ];

  beforeEach(deleteDatabase);

  afterEach(deleteDatabase);

  it('Save-load round trip', async () => {
    const testStartDate = new Date();
    const handler = tf.io.getSaveHandlers('indexeddb://FooModel')[0];

    const saveResult = await handler.save(artifacts1);
    expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
        .toBeGreaterThanOrEqual(testStartDate.getTime());
    // Note: The following two assertions work only because there is no
    //   non-ASCII characters in `modelTopology1` and `weightSpecs1`.
    expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
        .toEqual(JSON.stringify(modelTopology1).length);
    expect(saveResult.modelArtifactsInfo.weightSpecsBytes)
        .toEqual(JSON.stringify(weightSpecs1).length);
    expect(saveResult.modelArtifactsInfo.weightDataBytes)
        .toEqual(weightData1.byteLength);

    const loadedArtifacts = await handler.load();
    expect(loadedArtifacts.modelTopology).toEqual(modelTopology1);
    expect(loadedArtifacts.weightSpecs).toEqual(weightSpecs1);
    expect(loadedArtifacts.format).toEqual('layers-model');
    expect(loadedArtifacts.generatedBy).toEqual('TensorFlow.js v0.0.0');
    expect(loadedArtifacts.convertedBy).toEqual(null);
    expect(loadedArtifacts.modelInitializer).toEqual({});
    expectArrayBuffersEqual(loadedArtifacts.weightData, weightData1);
  });

  it('Save two models and load one', async () => {
    const weightData2 = new ArrayBuffer(24);
    const artifacts2: tf.io.ModelArtifacts = {
      modelTopology: modelTopology1,
      weightSpecs: weightSpecs2,
      weightData: weightData2,
    };
    const handler1 = tf.io.getSaveHandlers('indexeddb://Model/1')[0];
    const saveResult1 = await handler1.save(artifacts1);
    // Note: The following two assertions work only because there is no
    // non-ASCII characters in `modelTopology1` and `weightSpecs1`.
    expect(saveResult1.modelArtifactsInfo.modelTopologyBytes)
        .toEqual(JSON.stringify(modelTopology1).length);
    expect(saveResult1.modelArtifactsInfo.weightSpecsBytes)
        .toEqual(JSON.stringify(weightSpecs1).length);
    expect(saveResult1.modelArtifactsInfo.weightDataBytes)
        .toEqual(weightData1.byteLength);

    const handler2 = tf.io.getSaveHandlers('indexeddb://Model/2')[0];
    const saveResult2 = await handler2.save(artifacts2);
    expect(saveResult2.modelArtifactsInfo.dateSaved.getTime())
        .toBeGreaterThanOrEqual(
            saveResult1.modelArtifactsInfo.dateSaved.getTime());
    // Note: The following two assertions work only because there is
    // no non-ASCII characters in `modelTopology1` and
    // `weightSpecs1`.
    expect(saveResult2.modelArtifactsInfo.modelTopologyBytes)
        .toEqual(JSON.stringify(modelTopology1).length);
    expect(saveResult2.modelArtifactsInfo.weightSpecsBytes)
        .toEqual(JSON.stringify(weightSpecs2).length);
    expect(saveResult2.modelArtifactsInfo.weightDataBytes)
        .toEqual(weightData2.byteLength);

    const loadedArtifacts = await handler1.load();
    expect(loadedArtifacts.modelTopology).toEqual(modelTopology1);
    expect(loadedArtifacts.weightSpecs).toEqual(weightSpecs1);
    expectArrayBuffersEqual(loadedArtifacts.weightData, weightData1);
  });

  it('Loading nonexistent model fails', async () => {
    const handler = tf.io.getSaveHandlers('indexeddb://NonexistentModel')[0];

    try {
      await handler.load();
      fail('Loading nonexistent model from IndexedDB succeeded unexpectly');
    } catch (err) {
      expect(err.message)
          .toEqual(
              'Cannot find model with path \'NonexistentModel\' in IndexedDB.');
    }
  });

  it('Null, undefined or empty modelPath throws Error', () => {
    expect(() => browserIndexedDB(null))
        .toThrowError(
            /IndexedDB, modelPath must not be null, undefined or empty/);
    expect(() => browserIndexedDB(undefined))
        .toThrowError(
            /IndexedDB, modelPath must not be null, undefined or empty/);
    expect(() => browserIndexedDB(''))
        .toThrowError(
            /IndexedDB, modelPath must not be null, undefined or empty./);
  });

  it('router', () => {
    expect(indexedDBRouter('indexeddb://bar') instanceof BrowserIndexedDB)
        .toEqual(true);
    expect(indexedDBRouter('localstorage://bar')).toBeNull();
    expect(indexedDBRouter('qux')).toBeNull();
  });

  it('Manager: List models: 0 result', async () => {
    // Before any model is saved, listModels should return empty result.
    const models = await new BrowserIndexedDBManager().listModels();
    expect(models).toEqual({});
  });

  it('Manager: List models: 1 result', async () => {
    const handler = tf.io.getSaveHandlers('indexeddb://baz/QuxModel')[0];
    const saveResult = await handler.save(artifacts1);

    // After successful saving, there should be one model.
    const models = await new BrowserIndexedDBManager().listModels();
    expect(Object.keys(models).length).toEqual(1);
    expect(models['baz/QuxModel'].modelTopologyType)
        .toEqual(saveResult.modelArtifactsInfo.modelTopologyType);
    expect(models['baz/QuxModel'].modelTopologyBytes)
        .toEqual(saveResult.modelArtifactsInfo.modelTopologyBytes);
    expect(models['baz/QuxModel'].weightSpecsBytes)
        .toEqual(saveResult.modelArtifactsInfo.weightSpecsBytes);
    expect(models['baz/QuxModel'].weightDataBytes)
        .toEqual(saveResult.modelArtifactsInfo.weightDataBytes);
  });

  it('Manager: List models: 2 results', async () => {
    // First, save a model.
    const handler1 = tf.io.getSaveHandlers('indexeddb://QuxModel')[0];
    const saveResult1 = await handler1.save(artifacts1);

    // Then, save the model under another path.
    const handler2 = tf.io.getSaveHandlers('indexeddb://repeat/QuxModel')[0];
    const saveResult2 = await handler2.save(artifacts1);

    // After successful saving, there should be two models.
    const models = await new BrowserIndexedDBManager().listModels();
    expect(Object.keys(models).length).toEqual(2);
    expect(models['QuxModel'].modelTopologyType)
        .toEqual(saveResult1.modelArtifactsInfo.modelTopologyType);
    expect(models['QuxModel'].modelTopologyBytes)
        .toEqual(saveResult1.modelArtifactsInfo.modelTopologyBytes);
    expect(models['QuxModel'].weightSpecsBytes)
        .toEqual(saveResult1.modelArtifactsInfo.weightSpecsBytes);
    expect(models['QuxModel'].weightDataBytes)
        .toEqual(saveResult1.modelArtifactsInfo.weightDataBytes);
    expect(models['repeat/QuxModel'].modelTopologyType)
        .toEqual(saveResult2.modelArtifactsInfo.modelTopologyType);
    expect(models['repeat/QuxModel'].modelTopologyBytes)
        .toEqual(saveResult2.modelArtifactsInfo.modelTopologyBytes);
    expect(models['repeat/QuxModel'].weightSpecsBytes)
        .toEqual(saveResult2.modelArtifactsInfo.weightSpecsBytes);
    expect(models['repeat/QuxModel'].weightDataBytes)
        .toEqual(saveResult2.modelArtifactsInfo.weightDataBytes);
  });

  it('Manager: Successful removeModel', async () => {
    // First, save a model.
    const handler1 = tf.io.getSaveHandlers('indexeddb://QuxModel')[0];
    await handler1.save(artifacts1);

    // Then, save the model under another path.
    const handler2 = tf.io.getSaveHandlers('indexeddb://repeat/QuxModel')[0];
    await handler2.save(artifacts1);

    // After successful saving, delete the first save, and then
    // `listModel` should give only one result.
    const manager = new BrowserIndexedDBManager();
    await manager.removeModel('QuxModel');

    const models = await manager.listModels();
    expect(Object.keys(models)).toEqual(['repeat/QuxModel']);
  });

  it('Manager: Successful removeModel with URL scheme', async () => {
    // First, save a model.
    const handler1 = tf.io.getSaveHandlers('indexeddb://QuxModel')[0];
    await handler1.save(artifacts1);

    // Then, save the model under another path.
    const handler2 = tf.io.getSaveHandlers('indexeddb://repeat/QuxModel')[0];
    await handler2.save(artifacts1);

    // After successful saving, delete the first save, and then
    // `listModel` should give only one result.
    const manager = new BrowserIndexedDBManager();

    // Delete a model specified with a path that includes the
    // indexeddb:// scheme prefix should work.
    manager.removeModel('indexeddb://QuxModel');

    const models = await manager.listModels();
    expect(Object.keys(models)).toEqual(['repeat/QuxModel']);
  });

  it('Manager: Failed removeModel', async () => {
    try {
      // Attempt to delete a nonexistent model is expected to fail.
      await new BrowserIndexedDBManager().removeModel('nonexistent');
      fail('Deleting nonexistent model succeeded unexpectedly.');
    } catch (err) {
      expect(err.message)
          .toEqual('Cannot find model with path \'nonexistent\' in IndexedDB.');
    }
  });
});
