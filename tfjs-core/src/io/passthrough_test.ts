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
 * Unit tests for passthrough IOHandlers.
 */

import * as tf from '../index';
import {BROWSER_ENVS, describeWithFlags} from '../jasmine_util';

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
};

describeWithFlags('Passthrough Saver', BROWSER_ENVS, () => {
  it('passes provided arguments through on save', async () => {
    const testStartDate = new Date();
    let savedArtifacts: tf.io.ModelArtifacts = null;

    async function saveHandler(artifacts: tf.io.ModelArtifacts):
        Promise<tf.io.SaveResult> {
      savedArtifacts = artifacts;
      return {
        modelArtifactsInfo: {
          dateSaved: testStartDate,
          modelTopologyType: 'JSON',
          modelTopologyBytes: JSON.stringify(modelTopology1).length,
          weightSpecsBytes: JSON.stringify(weightSpecs1).length,
          weightDataBytes: weightData1.byteLength,
        }
      };
    }

    const saveTrigger = tf.io.withSaveHandler(saveHandler);
    const saveResult = await saveTrigger.save(artifacts1);

    expect(saveResult.errors).toEqual(undefined);
    const artifactsInfo = saveResult.modelArtifactsInfo;
    expect(artifactsInfo.dateSaved.getTime())
        .toBeGreaterThanOrEqual(testStartDate.getTime());
    expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
        .toEqual(JSON.stringify(modelTopology1).length);
    expect(saveResult.modelArtifactsInfo.weightSpecsBytes)
        .toEqual(JSON.stringify(weightSpecs1).length);
    expect(saveResult.modelArtifactsInfo.weightDataBytes).toEqual(16);

    expect(savedArtifacts.modelTopology).toEqual(modelTopology1);
    expect(savedArtifacts.weightSpecs).toEqual(weightSpecs1);
    expect(savedArtifacts.weightData).toEqual(weightData1);
  });
});

describeWithFlags('Passthrough Loader', BROWSER_ENVS, () => {
  it('load topology and weights: legacy signature', async () => {
    const passthroughHandler =
        tf.io.fromMemory(modelTopology1, weightSpecs1, weightData1);
    const modelArtifacts = await passthroughHandler.load();
    expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
    expect(modelArtifacts.weightSpecs).toEqual(weightSpecs1);
    expect(modelArtifacts.weightData).toEqual(weightData1);
    expect(modelArtifacts.userDefinedMetadata).toEqual(undefined);
  });

  it('load topology and weights', async () => {
    const passthroughHandler = tf.io.fromMemory({
      modelTopology: modelTopology1,
      weightSpecs: weightSpecs1,
      weightData: weightData1
    });
    const modelArtifacts = await passthroughHandler.load();
    expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
    expect(modelArtifacts.weightSpecs).toEqual(weightSpecs1);
    expect(modelArtifacts.weightData).toEqual(weightData1);
    expect(modelArtifacts.userDefinedMetadata).toEqual(undefined);
  });

  it('load model topology only: legacy signature', async () => {
    const passthroughHandler = tf.io.fromMemory(modelTopology1);
    const modelArtifacts = await passthroughHandler.load();
    expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
    expect(modelArtifacts.weightSpecs).toEqual(undefined);
    expect(modelArtifacts.weightData).toEqual(undefined);
    expect(modelArtifacts.userDefinedMetadata).toEqual(undefined);
  });

  it('load model topology only', async () => {
    const passthroughHandler =
        tf.io.fromMemory({modelTopology: modelTopology1});
    const modelArtifacts = await passthroughHandler.load();
    expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
    expect(modelArtifacts.weightSpecs).toEqual(undefined);
    expect(modelArtifacts.weightData).toEqual(undefined);
    expect(modelArtifacts.userDefinedMetadata).toEqual(undefined);
  });

  it('load topology, weights, and user-defined metadata', async () => {
    const userDefinedMetadata: {} = {'fooField': 'fooValue'};
    const passthroughHandler = tf.io.fromMemory({
      modelTopology: modelTopology1,
      weightSpecs: weightSpecs1,
      weightData: weightData1,
      userDefinedMetadata
    });
    const modelArtifacts = await passthroughHandler.load();
    expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
    expect(modelArtifacts.weightSpecs).toEqual(weightSpecs1);
    expect(modelArtifacts.weightData).toEqual(weightData1);
    expect(modelArtifacts.userDefinedMetadata).toEqual(userDefinedMetadata);
  });
});
