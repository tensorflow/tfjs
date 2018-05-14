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
 * Unit tests for browser_http.ts.
 */

import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
import {CPU_ENVS} from '../test_util';
import {BrowserHTTPRequest, httpRequestRouter} from './browser_http';

describeWithFlags('browserHTTPRequest', CPU_ENVS, () => {
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

  let requestInits: RequestInit[] = [];

  beforeEach(() => {
    requestInits = [];
    spyOn(window, 'fetch').and.callFake((path: string, init: RequestInit) => {
      if (path === 'model-upload-test' || path === 'http://model-upload-test') {
        requestInits.push(init);
        return new Response(null, {status: 200});
      } else {
        return new Response(null, {status: 404});
      }
    });
  });

  it('Save topology and weights, default POST method', done => {
    const testStartDate = new Date();
    const handler = tf.io.getSaveHandlers('http://model-upload-test')[0];
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
          expect(saveResult.modelArtifactsInfo.weightDataBytes)
              .toEqual(weightData1.byteLength);

          expect(requestInits.length).toEqual(1);
          const init = requestInits[0];
          expect(init.method).toEqual('POST');
          const body = init.body as FormData;
          const jsonFile = body.get('model.json') as File;
          const jsonFileReader = new FileReader();
          jsonFileReader.onload = (event: Event) => {
            // tslint:disable-next-line:no-any
            const modelJSON = JSON.parse((event.target as any).result);
            expect(modelJSON.modelTopology).toEqual(modelTopology1);
            expect(modelJSON.weightsManifest.length).toEqual(1);
            expect(modelJSON.weightsManifest[0].weights).toEqual(weightSpecs1);

            const weightsFile = body.get('model.weights.bin') as File;
            const weightsFileReader = new FileReader();
            weightsFileReader.onload = (event: Event) => {
              // tslint:disable-next-line:no-any
              const weightData = (event.target as any).result as ArrayBuffer;
              expect(new Uint8Array(weightData))
                  .toEqual(new Uint8Array(weightData1));
              done();
            };
            weightsFileReader.onerror = (error: ErrorEvent) => {
              done.fail(error.message);
            };
            weightsFileReader.readAsArrayBuffer(weightsFile);
          };
          jsonFileReader.onerror = (error: ErrorEvent) => {
            done.fail(error.message);
          };
          jsonFileReader.readAsText(jsonFile);
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Save topology only, default POST method', done => {
    const testStartDate = new Date();
    const handler = tf.io.getSaveHandlers('http://model-upload-test')[0];
    const topologyOnlyArtifacts = {modelTopology: modelTopology1};
    handler.save(topologyOnlyArtifacts)
        .then(saveResult => {
          expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
              .toBeGreaterThanOrEqual(testStartDate.getTime());
          // Note: The following two assertions work only because there is no
          //   non-ASCII characters in `modelTopology1` and `weightSpecs1`.
          expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
              .toEqual(JSON.stringify(modelTopology1).length);
          expect(saveResult.modelArtifactsInfo.weightSpecsBytes).toEqual(0);
          expect(saveResult.modelArtifactsInfo.weightDataBytes).toEqual(0);

          expect(requestInits.length).toEqual(1);
          const init = requestInits[0];
          expect(init.method).toEqual('POST');
          const body = init.body as FormData;
          const jsonFile = body.get('model.json') as File;
          const jsonFileReader = new FileReader();
          jsonFileReader.onload = (event: Event) => {
            // tslint:disable-next-line:no-any
            const modelJSON = JSON.parse((event.target as any).result);
            expect(modelJSON.modelTopology).toEqual(modelTopology1);
            // No weights should have been sent to the server.
            expect(body.get('model.weights.bin')).toEqual(null);
            done();
          };
          jsonFileReader.onerror = (error: ErrorEvent) => {
            done.fail(error.message);
          };
          jsonFileReader.readAsText(jsonFile);
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Save topology and weights, PUT method, extra headers', done => {
    const testStartDate = new Date();
    const handler = tf.io.browserHTTPRequest('model-upload-test', {
      method: 'PUT',
      headers: {
        'header_key_1': 'header_value_1',
        'header_key_2': 'header_value_2'
      }
    });
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
          expect(saveResult.modelArtifactsInfo.weightDataBytes)
              .toEqual(weightData1.byteLength);

          expect(requestInits.length).toEqual(1);
          const init = requestInits[0];
          expect(init.method).toEqual('PUT');

          // Check headers.
          expect(init.headers).toEqual({
            'header_key_1': 'header_value_1',
            'header_key_2': 'header_value_2'
          });

          const body = init.body as FormData;
          const jsonFile = body.get('model.json') as File;
          const jsonFileReader = new FileReader();
          jsonFileReader.onload = (event: Event) => {
            // tslint:disable-next-line:no-any
            const modelJSON = JSON.parse((event.target as any).result);
            expect(modelJSON.modelTopology).toEqual(modelTopology1);
            expect(modelJSON.weightsManifest.length).toEqual(1);
            expect(modelJSON.weightsManifest[0].weights).toEqual(weightSpecs1);

            const weightsFile = body.get('model.weights.bin') as File;
            const weightsFileReader = new FileReader();
            weightsFileReader.onload = (event: Event) => {
              // tslint:disable-next-line:no-any
              const weightData = (event.target as any).result as ArrayBuffer;
              expect(new Uint8Array(weightData))
                  .toEqual(new Uint8Array(weightData1));
              done();
            };
            weightsFileReader.onerror = (error: ErrorEvent) => {
              done.fail(error.message);
            };
            weightsFileReader.readAsArrayBuffer(weightsFile);
          };
          jsonFileReader.onerror = (error: ErrorEvent) => {
            done.fail(error.message);
          };
          jsonFileReader.readAsText(jsonFile);
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('404 response causes Error', done => {
    const handler = tf.io.getSaveHandlers('http://invalid/path')[0];
    handler.save(artifacts1)
        .then(saveResult => {
          done.fail(
              'Calling browserHTTPRequest at invalid URL succeeded ' +
              'unexpectedly');
        })
        .catch(err => {
          done();
        });
  });

  it('Existing body leads to Error', () => {
    const key1Data = '1337';
    const key2Data = '42';
    const extraFormData = new FormData();
    extraFormData.set('key1', key1Data);
    extraFormData.set('key2', key2Data);
    expect(() => tf.io.browserHTTPRequest('model-upload-test', {
      body: extraFormData
    })).toThrowError(/requestInit is expected to have no pre-existing body/);
  });

  it('Empty, null or undefined URL paths lead to Error', () => {
    expect(() => tf.io.browserHTTPRequest(null))
        .toThrowError(/must not be null, undefined or empty/);
    expect(() => tf.io.browserHTTPRequest(undefined))
        .toThrowError(/must not be null, undefined or empty/);
    expect(() => tf.io.browserHTTPRequest(''))
        .toThrowError(/must not be null, undefined or empty/);
  });

  it('router', () => {
    expect(httpRequestRouter('http://bar/foo') instanceof BrowserHTTPRequest)
        .toEqual(true);
    expect(
        httpRequestRouter('https://localhost:5000/upload') instanceof
        BrowserHTTPRequest)
        .toEqual(true);
    expect(httpRequestRouter('localhost://foo')).toBeNull();
    expect(httpRequestRouter('foo:5000/bar')).toBeNull();
  });
});
