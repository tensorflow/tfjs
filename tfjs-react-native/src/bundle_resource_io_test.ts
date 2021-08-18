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
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {bundleResourceIO} from './bundle_resource_io';
import * as tfjsRn from './platform_react_native';
import {RN_ENVS} from './test_env_registry';

describeWithFlags('BundleResourceIO', RN_ENVS, () => {
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

  it('constructs an IOHandler', async () => {
    const modelJson: tf.io.ModelJSON = {
      modelTopology: modelTopology1,
      weightsManifest: [{
        paths: [],
        weights: weightSpecs1,
      }]

    };
    const resourceId = 1;
    const handler = bundleResourceIO(modelJson, resourceId);
    expect(typeof handler.load).toBe('function');
    expect(typeof handler.save).toBe('function');
  });

  it('loads model artifacts', async () => {
    const response = new Response(weightData1);
    spyOn(tfjsRn, 'fetch').and.returnValue(Promise.resolve(response));

    const modelJson: tf.io.ModelJSON = {
      modelTopology: modelTopology1,
      weightsManifest: [{
        paths: [],
        weights: weightSpecs1,
      }]

    };
    const resourceId = 1;
    const handler = bundleResourceIO(modelJson, resourceId);

    const loaded = await handler.load();

    expect(loaded.modelTopology).toEqual(modelTopology1);
    expect(loaded.weightSpecs).toEqual(weightSpecs1);
    expect(loaded.weightData).toEqual(weightData1);
  });

  it('errors on string modelJSON', async () => {
    const response = new Response(weightData1);
    spyOn(tf.env().platform, 'fetch')
        .and.returnValue(Promise.resolve(response));

    const modelJson = `{
      modelTopology: modelTopology1,
      weightsManifest: [{
        paths: [],
        weights: weightSpecs1,
      }]
    }`;
    const resourceId = 1;
    expect(
        () => bundleResourceIO(
            modelJson as unknown as tf.io.ModelJSON, resourceId))
        .toThrow(new Error(
            'modelJson must be a JavaScript object (and not a string).\n' +
            'Have you wrapped yor asset path in a require() statment?'));
  });
});

describeWithFlags('BundleResourceIO Sharded', RN_ENVS, () => {
  // Test data.
  const modelTopology: {} = {
    'class_name': 'Sequential',
    'keras_version': '2.1.4',
    'config': [
      {
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
      },
      {
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
          'name': 'dense2',
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
      }
    ],
    'backend': 'tensorflow'
  };

  const weightSpecs: tf.io.WeightsManifestEntry[] = [
    {
      name: 'dense/kernel',
      shape: [3, 1],
      dtype: 'float32',
    },
    {
      name: 'dense/bias',
      shape: [1],
      dtype: 'float32',
    },
    {
      name: 'dense2/kernel',
      shape: [3, 1],
      dtype: 'float32',
    },
    {
      name: 'dense2/bias',
      shape: [1],
      dtype: 'float32',
    }
  ];
  const weightData1 = new ArrayBuffer(16);
  const weightData2 = new ArrayBuffer(16);
  const resourceIds = [1, 2];

  const combinedWeightsExpected = new ArrayBuffer(32);

  it('constructs an IOHandler', async () => {
    const modelJson: tf.io.ModelJSON = {
      modelTopology,
      weightsManifest: [{
        paths: [],
        weights: weightSpecs,
      }]

    };

    const handler = bundleResourceIO(modelJson, resourceIds);
    expect(typeof handler.load).toBe('function');
    expect(typeof handler.save).toBe('function');
  });

  it('loads model artifacts', async () => {
    spyOn(tf.env().platform, 'fetch')
        .and.returnValues(
            Promise.resolve(new Response(weightData1)),
            Promise.resolve(new Response(weightData2)),
        );

    const modelJson: tf.io.ModelJSON = {
      modelTopology,
      weightsManifest: [{
        paths: [],
        weights: weightSpecs,
      }]

    };

    const handler = bundleResourceIO(modelJson, resourceIds);

    const loaded = await handler.load();

    expect(loaded.modelTopology).toEqual(modelTopology);
    expect(loaded.weightSpecs).toEqual(weightSpecs);
    expect(loaded.weightData).toEqual(combinedWeightsExpected);
  });
});
