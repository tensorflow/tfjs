/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '../index';

import {arrayBufferToBase64String, base64StringToArrayBuffer} from './io_utils';
import {purgeLocalStorageArtifacts} from './local_storage';

describe('LocalStorage', () => {
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
  };

  function findOverflowingByteSize(): number {
    const LS = window.localStorage;
    const probeKey =
        `tfjs_test_probe_values_${new Date().getTime()}_${Math.random()}`;
    const minKilobytes = 200;
    const stepKilobytes = 200;
    const maxKilobytes = 40000;
    for (let kilobytes = minKilobytes; kilobytes < maxKilobytes;
         kilobytes += stepKilobytes) {
      const bytes = kilobytes * 1024;
      const data = new ArrayBuffer(bytes);
      try {
        const encoded = arrayBufferToBase64String(data);
        LS.setItem(probeKey, encoded);
      } catch (err) {
        return bytes;
      }
      LS.removeItem(probeKey);
    }
    throw new Error(
        `Unable to determined overflowing byte size up to ${maxKilobytes} kB.`);
  }

  beforeEach(() => {
    purgeLocalStorageArtifacts();
  });

  afterEach(() => {
    purgeLocalStorageArtifacts();
  });

  it('Save artifacts succeeds', async done => {
    const testStartDate = new Date();
    const handler = tf.io.browserLocalStorage('FooModel');
    handler.save(artifacts1)
        .then(saveResult => {
          expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
              .toBeGreaterThanOrEqual(testStartDate.getTime());
          // Note: The following two assertions work only because there is no
          //   non-ASCII characters in `modelTopology1` and `weightSpecs1`.
          expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
              .toEqual(JSON.stringify(modelTopology1).length);
          expect(saveResult.modelArtifactsInfo.weightSpecsBytes)
              .toEqual(JSON.stringify(weightSpecs1).length);
          expect(saveResult.modelArtifactsInfo.weightDataBytes).toEqual(16);

          // Check the content of the saved items in local storage.
          const LS = window.localStorage;
          const info =
              JSON.parse(LS.getItem('tensorflowjs_models/FooModel/info'));
          expect(Date.parse(info.dateSaved))
              .toEqual(saveResult.modelArtifactsInfo.dateSaved.getTime());
          expect(info.modelTopologyBytes)
              .toEqual(saveResult.modelArtifactsInfo.modelTopologyBytes);
          expect(info.weightSpecsBytes)
              .toEqual(saveResult.modelArtifactsInfo.weightSpecsBytes);
          expect(info.weightDataBytes)
              .toEqual(saveResult.modelArtifactsInfo.weightDataBytes);

          const topologyString =
              LS.getItem('tensorflowjs_models/FooModel/model_topology');
          expect(JSON.stringify(modelTopology1)).toEqual(topologyString);

          const weightSpecsString =
              LS.getItem('tensorflowjs_models/FooModel/weight_specs');
          expect(JSON.stringify(weightSpecs1)).toEqual(weightSpecsString);

          const weightDataBase64String =
              LS.getItem('tensorflowjs_models/FooModel/weight_data');
          expect(base64StringToArrayBuffer(weightDataBase64String))
              .toEqual(weightData1);

          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Save-load round trip succeeds', async done => {
    const handler1 = tf.io.browserLocalStorage('FooModel');
    handler1.save(artifacts1)
        .then(saveResult => {
          const handler2 = tf.io.browserLocalStorage('FooModel');
          handler2.load()
              .then(loaded => {
                expect(loaded.modelTopology).toEqual(modelTopology1);
                expect(loaded.weightSpecs).toEqual(weightSpecs1);
                expect(loaded.weightData).toEqual(weightData1);
                done();
              })
              .catch(err => {
                console.error(err.stack);
              });
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Loading nonexistent model fails.', async done => {
    const handler = tf.io.browserLocalStorage('NonexistentModel');
    handler.load()
        .then(aritfacts => {
          fail('Loading nonexistent model succeeded unexpectedly.');
        })
        .catch(err => {
          expect(err.message)
              .toEqual(
                  'In local storage, there is no model with name ' +
                  '\'NonexistentModel\'');
          done();
        });
  });

  it('Loading model with missing topology fails.', async done => {
    const handler1 = tf.io.browserLocalStorage('FooModel');
    handler1.save(artifacts1)
        .then(saveResult => {
          // Manually remove the topology item from local storage.
          window.localStorage.removeItem(
              'tensorflowjs_models/FooModel/model_topology');

          const handler2 = tf.io.browserLocalStorage('FooModel');
          handler2.load()
              .then(aritfacts => {
                fail(
                    'Loading of model with missing topology succeeded ' +
                    'unexpectedly.');
              })
              .catch(err => {
                expect(err.message)
                    .toEqual(
                        'In local storage, the topology of model ' +
                        '\'FooModel\' is missing.');
                done();
              });
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Loading model with missing weight specs fails.', async done => {
    const handler1 = tf.io.browserLocalStorage('FooModel');
    handler1.save(artifacts1)
        .then(saveResult => {
          // Manually remove the weight specs item from local storage.
          window.localStorage.removeItem(
              'tensorflowjs_models/FooModel/weight_specs');

          const handler2 = tf.io.browserLocalStorage('FooModel');
          handler2.load()
              .then(aritfacts => {
                fail(
                    'Loading of model with missing weight specs succeeded ' +
                    'unexpectedly.');
              })
              .catch(err => {
                expect(err.message)
                    .toEqual(
                        'In local storage, the weight specs of model ' +
                        '\'FooModel\' are missing.');
                done();
              });
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Loading model with missing weight data fails.', async done => {
    const handler1 = tf.io.browserLocalStorage('FooModel');
    handler1.save(artifacts1)
        .then(saveResult => {
          // Manually remove the weight data item from local storage.
          window.localStorage.removeItem(
              'tensorflowjs_models/FooModel/weight_data');

          const handler2 = tf.io.browserLocalStorage('FooModel');
          handler2.load()
              .then(aritfacts => {
                fail(
                    'Loading of model with missing weight data succeeded ' +
                    'unexpectedly.');
              })
              .catch(err => {
                expect(err.message)
                    .toEqual(
                        'In local storage, the binary weight values of model ' +
                        '\'FooModel\' are missing.');
                done();
              });
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Data size too large leads to error thrown', async done => {
    const overflowByteSize = findOverflowingByteSize();
    const overflowArtifacts: tf.io.ModelArtifacts = {
      modelTopology: modelTopology1,
      weightSpecs: weightSpecs1,
      weightData: new ArrayBuffer(overflowByteSize),
    };
    const handler1 = tf.io.browserLocalStorage('FooModel');
    handler1.save(overflowArtifacts)
        .then(saveResult => {
          fail(
              'Saving of model of overflowing-size weight data succeeded ' +
              'unexpectedly.');
        })
        .catch(err => {
          expect((err.message as string)
                     .indexOf(
                         'Failed to save model \'FooModel\' to local storage'))
              .toEqual(0);
          done();
        });
  });

  it('Null, undefined or empty modelPath throws Error', () => {
    expect(() => tf.io.browserLocalStorage(null))
        .toThrowError(
            /local storage, modelPath must not be null, undefined or empty/);
    expect(() => tf.io.browserLocalStorage(undefined))
        .toThrowError(
            /local storage, modelPath must not be null, undefined or empty/);
    expect(() => tf.io.browserLocalStorage(''))
        .toThrowError(
            /local storage, modelPath must not be null, undefined or empty./);
  });
});
