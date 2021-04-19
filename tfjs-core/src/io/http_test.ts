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

import * as tf from '../index';
import {BROWSER_ENVS, CHROME_ENVS, describeWithFlags, NODE_ENVS} from '../jasmine_util';
import {HTTPRequest, httpRouter, parseUrl} from './http';

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

let fetchSpy: jasmine.Spy;

type TypedArrays = Float32Array|Int32Array|Uint8Array|Uint16Array;
const fakeResponse =
    (body: string|TypedArrays|ArrayBuffer, contentType: string, path: string) =>
        ({
          ok: true,
          json() {
            return Promise.resolve(JSON.parse(body as string));
          },
          arrayBuffer() {
            const buf: ArrayBuffer = (body as TypedArrays).buffer ?
                (body as TypedArrays).buffer :
                body as ArrayBuffer;
            return Promise.resolve(buf);
          },
          headers: {get: (key: string) => contentType},
          url: path
        });

const setupFakeWeightFiles =
    (fileBufferMap: {
      [filename: string]: {
        data: string|Float32Array|Int32Array|ArrayBuffer|Uint8Array|Uint16Array,
        contentType: string
      }
    },
     requestInits: {[key: string]: RequestInit}) => {
      fetchSpy = spyOn(tf.env().platform, 'fetch')
                     .and.callFake((path: string, init: RequestInit) => {
                       if (fileBufferMap[path]) {
                         requestInits[path] = init;
                         return Promise.resolve(fakeResponse(
                             fileBufferMap[path].data,
                             fileBufferMap[path].contentType, path));
                       } else {
                         return Promise.reject('path not found');
                       }
                     });
    };

describeWithFlags('http-load fetch', NODE_ENVS, () => {
  let requestInits: {[key: string]: {headers: {[key: string]: string}}};
  // tslint:disable-next-line:no-any
  let originalFetch: any;
  // simulate a fetch polyfill, this needs to be non-null for spyOn to work
  beforeEach(() => {
    // tslint:disable-next-line:no-any
    originalFetch = (global as any).fetch;
    // tslint:disable-next-line:no-any
    (global as any).fetch = () => {};
    requestInits = {};
  });

  afterAll(() => {
    // tslint:disable-next-line:no-any
    (global as any).fetch = originalFetch;
  });

  it('1 group, 2 weights, 1 path', async () => {
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
          shape: [2],
          dtype: 'float32',
        }
      ]
    }];
    const floatData = new Float32Array([1, 3, 3, 7, 4]);
    setupFakeWeightFiles(
        {
          './model.json': {
            data: JSON.stringify({
              modelTopology: modelTopology1,
              weightsManifest: weightManifest1,
              format: 'tfjs-layers',
              generatedBy: '1.15',
              convertedBy: '1.3.1',
              signature: null,
              userDefinedMetadata: {}
            }),
            contentType: 'application/json'
          },
          './weightfile0':
              {data: floatData, contentType: 'application/octet-stream'},
        },
        requestInits);

    const handler = tf.io.http('./model.json');
    const modelArtifacts = await handler.load();
    expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
    expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
    expect(modelArtifacts.format).toEqual('tfjs-layers');
    expect(modelArtifacts.generatedBy).toEqual('1.15');
    expect(modelArtifacts.convertedBy).toEqual('1.3.1');
    expect(modelArtifacts.userDefinedMetadata).toEqual({});
    expect(new Float32Array(modelArtifacts.weightData)).toEqual(floatData);
  });

  it('throw exception if no fetch polyfill', () => {
    // tslint:disable-next-line:no-any
    delete (global as any).fetch;
    try {
      tf.io.http('./model.json');
    } catch (err) {
      expect(err.message).toMatch(/Unable to find fetch polyfill./);
    }
  });
});

// Turned off for other browsers due to:
// https://github.com/tensorflow/tfjs/issues/426
describeWithFlags('http-save', CHROME_ENVS, () => {
  // Test data.
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
    signature: null,
    userDefinedMetadata: {},
    modelInitializer: {}
  };

  let requestInits: RequestInit[] = [];

  beforeEach(() => {
    requestInits = [];
    spyOn(tf.env().platform, 'fetch')
        .and.callFake((path: string, init: RequestInit) => {
          if (path === 'model-upload-test' ||
              path === 'http://model-upload-test') {
            requestInits.push(init);
            return Promise.resolve(new Response(null, {status: 200}));
          } else {
            return Promise.reject(new Response(null, {status: 404}));
          }
        });
  });

  it('Save topology and weights, default POST method', (done) => {
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
            weightsFileReader.onerror = ev => {
              done.fail(weightsFileReader.error.message);
            };
            weightsFileReader.readAsArrayBuffer(weightsFile);
          };
          jsonFileReader.onerror = ev => {
            done.fail(jsonFileReader.error.message);
          };
          jsonFileReader.readAsText(jsonFile);
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Save topology only, default POST method', (done) => {
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
          jsonFileReader.onerror = event => {
            done.fail(jsonFileReader.error.message);
          };
          jsonFileReader.readAsText(jsonFile);
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Save topology and weights, PUT method, extra headers', (done) => {
    const testStartDate = new Date();
    const handler = tf.io.http('model-upload-test', {
      requestInit: {
        method: 'PUT',
        headers:
            {'header_key_1': 'header_value_1', 'header_key_2': 'header_value_2'}
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
            expect(modelJSON.format).toEqual('layers-model');
            expect(modelJSON.generatedBy).toEqual('TensorFlow.js v0.0.0');
            expect(modelJSON.convertedBy).toEqual(null);
            expect(modelJSON.modelTopology).toEqual(modelTopology1);
            expect(modelJSON.modelInitializer).toEqual({});
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
            weightsFileReader.onerror = event => {
              done.fail(weightsFileReader.error.message);
            };
            weightsFileReader.readAsArrayBuffer(weightsFile);
          };
          jsonFileReader.onerror = event => {
            done.fail(jsonFileReader.error.message);
          };
          jsonFileReader.readAsText(jsonFile);
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('404 response causes Error', (done) => {
    const handler = tf.io.getSaveHandlers('http://invalid/path')[0];
    handler.save(artifacts1)
        .then(saveResult => {
          done.fail(
              'Calling http at invalid URL succeeded ' +
              'unexpectedly');
        })
        .catch(err => {
          done();
        });
  });

  it('getLoadHandlers with one URL string', () => {
    const handlers = tf.io.getLoadHandlers('http://foo/model.json');
    expect(handlers.length).toEqual(1);
    expect(handlers[0] instanceof HTTPRequest).toEqual(true);
  });

  it('Existing body leads to Error', () => {
    expect(() => tf.io.http('model-upload-test', {
      requestInit: {body: 'existing body'}
    })).toThrowError(/requestInit is expected to have no pre-existing body/);
  });

  it('Empty, null or undefined URL paths lead to Error', () => {
    expect(() => tf.io.http(null))
        .toThrowError(/must not be null, undefined or empty/);
    expect(() => tf.io.http(undefined))
        .toThrowError(/must not be null, undefined or empty/);
    expect(() => tf.io.http(''))
        .toThrowError(/must not be null, undefined or empty/);
  });

  it('router', () => {
    expect(httpRouter('http://bar/foo') instanceof HTTPRequest).toEqual(true);
    expect(httpRouter('https://localhost:5000/upload') instanceof HTTPRequest)
        .toEqual(true);
    expect(httpRouter('localhost://foo')).toBeNull();
    expect(httpRouter('foo:5000/bar')).toBeNull();
  });
});

describeWithFlags('parseUrl', BROWSER_ENVS, () => {
  it('should parse url with no suffix', () => {
    const url = 'http://google.com/file';
    const [prefix, suffix] = parseUrl(url);
    expect(prefix).toEqual('http://google.com/');
    expect(suffix).toEqual('');
  });
  it('should parse url with suffix', () => {
    const url = 'http://google.com/file?param=1';
    const [prefix, suffix] = parseUrl(url);
    expect(prefix).toEqual('http://google.com/');
    expect(suffix).toEqual('?param=1');
  });
  it('should parse url with multiple serach params', () => {
    const url = 'http://google.com/a?x=1/file?param=1';
    const [prefix, suffix] = parseUrl(url);
    expect(prefix).toEqual('http://google.com/a?x=1/');
    expect(suffix).toEqual('?param=1');
  });
});

describeWithFlags('http-load', BROWSER_ENVS, () => {
  describe('JSON model', () => {
    let requestInits: {[key: string]: {headers: {[key: string]: string}}};

    beforeEach(() => {
      requestInits = {};
    });

    it('1 group, 2 weights, 1 path', async () => {
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
            shape: [2],
            dtype: 'float32',
          }
        ]
      }];
      const floatData = new Float32Array([1, 3, 3, 7, 4]);
      setupFakeWeightFiles(
          {
            './model.json': {
              data: JSON.stringify({
                modelTopology: modelTopology1,
                weightsManifest: weightManifest1,
                format: 'tfjs-graph-model',
                generatedBy: '1.15',
                convertedBy: '1.3.1',
                signature: null,
                userDefinedMetadata: {},
                modelInitializer: {}
              }),
              contentType: 'application/json'
            },
            './weightfile0':
                {data: floatData, contentType: 'application/octet-stream'},
          },
          requestInits);

      const handler = tf.io.http('./model.json');
      const modelArtifacts = await handler.load();
      expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
      expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
      expect(modelArtifacts.format).toEqual('tfjs-graph-model');
      expect(modelArtifacts.generatedBy).toEqual('1.15');
      expect(modelArtifacts.convertedBy).toEqual('1.3.1');
      expect(modelArtifacts.userDefinedMetadata).toEqual({});
      expect(modelArtifacts.modelInitializer).toEqual({});

      expect(new Float32Array(modelArtifacts.weightData)).toEqual(floatData);
      expect(Object.keys(requestInits).length).toEqual(2);
      // Assert that fetch is invoked with `window` as the context.
      expect(fetchSpy.calls.mostRecent().object).toEqual(window);
    });

    it('1 group, 2 weights, 1 path, with requestInit', async () => {
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
            shape: [2],
            dtype: 'float32',
          }
        ]
      }];
      const floatData = new Float32Array([1, 3, 3, 7, 4]);
      setupFakeWeightFiles(
          {
            './model.json': {
              data: JSON.stringify({
                modelTopology: modelTopology1,
                weightsManifest: weightManifest1
              }),
              contentType: 'application/json'
            },
            './weightfile0':
                {data: floatData, contentType: 'application/octet-stream'},
          },
          requestInits);

      const handler = tf.io.http(
          './model.json',
          {requestInit: {headers: {'header_key_1': 'header_value_1'}}});
      const modelArtifacts = await handler.load();
      expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
      expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
      expect(new Float32Array(modelArtifacts.weightData)).toEqual(floatData);
      expect(Object.keys(requestInits).length).toEqual(2);
      expect(Object.keys(requestInits).length).toEqual(2);
      expect(requestInits['./model.json'].headers['header_key_1'])
          .toEqual('header_value_1');
      expect(requestInits['./weightfile0'].headers['header_key_1'])
          .toEqual('header_value_1');

      expect(fetchSpy.calls.mostRecent().object).toEqual(window);
    });

    it('1 group, 2 weight, 2 paths', async () => {
      const weightManifest1: tf.io.WeightsManifestConfig = [{
        paths: ['weightfile0', 'weightfile1'],
        weights: [
          {
            name: 'dense/kernel',
            shape: [3, 1],
            dtype: 'float32',
          },
          {
            name: 'dense/bias',
            shape: [2],
            dtype: 'float32',
          }
        ]
      }];
      const floatData1 = new Float32Array([1, 3, 3]);
      const floatData2 = new Float32Array([7, 4]);
      setupFakeWeightFiles(
          {
            './model.json': {
              data: JSON.stringify({
                modelTopology: modelTopology1,
                weightsManifest: weightManifest1
              }),
              contentType: 'application/json'
            },
            './weightfile0':
                {data: floatData1, contentType: 'application/octet-stream'},
            './weightfile1':
                {data: floatData2, contentType: 'application/octet-stream'}
          },
          requestInits);

      const handler = tf.io.http('./model.json');
      const modelArtifacts = await handler.load();
      expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
      expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
      expect(new Float32Array(modelArtifacts.weightData))
          .toEqual(new Float32Array([1, 3, 3, 7, 4]));
    });

    it('2 groups, 2 weight, 2 paths', async () => {
      const weightsManifest: tf.io.WeightsManifestConfig = [
        {
          paths: ['weightfile0'],
          weights: [{
            name: 'dense/kernel',
            shape: [3, 1],
            dtype: 'float32',
          }]
        },
        {
          paths: ['weightfile1'],
          weights: [{
            name: 'dense/bias',
            shape: [2],
            dtype: 'float32',
          }],
        }
      ];
      const floatData1 = new Float32Array([1, 3, 3]);
      const floatData2 = new Float32Array([7, 4]);
      setupFakeWeightFiles(
          {
            './model.json': {
              data: JSON.stringify(
                  {modelTopology: modelTopology1, weightsManifest}),
              contentType: 'application/json'
            },
            './weightfile0':
                {data: floatData1, contentType: 'application/octet-stream'},
            './weightfile1':
                {data: floatData2, contentType: 'application/octet-stream'}
          },
          requestInits);

      const handler = tf.io.http('./model.json');
      const modelArtifacts = await handler.load();
      expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
      expect(modelArtifacts.weightSpecs)
          .toEqual(
              weightsManifest[0].weights.concat(weightsManifest[1].weights));
      expect(new Float32Array(modelArtifacts.weightData))
          .toEqual(new Float32Array([1, 3, 3, 7, 4]));
    });

    it('2 groups, 2 weight, 2 paths, Int32 and Uint8 Data', async () => {
      const weightsManifest: tf.io.WeightsManifestConfig = [
        {
          paths: ['weightfile0'],
          weights: [{
            name: 'fooWeight',
            shape: [3, 1],
            dtype: 'int32',
          }]
        },
        {
          paths: ['weightfile1'],
          weights: [{
            name: 'barWeight',
            shape: [2],
            dtype: 'bool',
          }],
        }
      ];
      const floatData1 = new Int32Array([1, 3, 3]);
      const floatData2 = new Uint8Array([7, 4]);
      setupFakeWeightFiles(
          {
            'path1/model.json': {
              data: JSON.stringify(
                  {modelTopology: modelTopology1, weightsManifest}),
              contentType: 'application/json'
            },
            'path1/weightfile0':
                {data: floatData1, contentType: 'application/octet-stream'},
            'path1/weightfile1':
                {data: floatData2, contentType: 'application/octet-stream'}
          },
          requestInits);

      const handler = tf.io.http('path1/model.json');
      const modelArtifacts = await handler.load();
      expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
      expect(modelArtifacts.weightSpecs)
          .toEqual(
              weightsManifest[0].weights.concat(weightsManifest[1].weights));
      expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
          .toEqual(new Int32Array([1, 3, 3]));
      expect(new Uint8Array(modelArtifacts.weightData.slice(12, 14)))
          .toEqual(new Uint8Array([7, 4]));
    });

    it('topology only', async () => {
      setupFakeWeightFiles(
          {
            './model.json': {
              data: JSON.stringify({modelTopology: modelTopology1}),
              contentType: 'application/json'
            },
          },
          requestInits);

      const handler = tf.io.http('./model.json');
      const modelArtifacts = await handler.load();
      expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
      expect(modelArtifacts.weightSpecs).toBeUndefined();
      expect(modelArtifacts.weightData).toBeUndefined();
    });

    it('weights only', async () => {
      const weightsManifest: tf.io.WeightsManifestConfig = [
        {
          paths: ['weightfile0'],
          weights: [{
            name: 'fooWeight',
            shape: [3, 1],
            dtype: 'int32',
          }]
        },
        {
          paths: ['weightfile1'],
          weights: [{
            name: 'barWeight',
            shape: [2],
            dtype: 'float32',
          }],
        }
      ];
      const floatData1 = new Int32Array([1, 3, 3]);
      const floatData2 = new Float32Array([-7, -4]);
      setupFakeWeightFiles(
          {
            'path1/model.json': {
              data: JSON.stringify({weightsManifest}),
              contentType: 'application/json'
            },
            'path1/weightfile0':
                {data: floatData1, contentType: 'application/octet-stream'},
            'path1/weightfile1':
                {data: floatData2, contentType: 'application/octet-stream'}
          },
          requestInits);

      const handler = tf.io.http('path1/model.json');
      const modelArtifacts = await handler.load();
      expect(modelArtifacts.modelTopology).toBeUndefined();
      expect(modelArtifacts.weightSpecs)
          .toEqual(
              weightsManifest[0].weights.concat(weightsManifest[1].weights));
      expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
          .toEqual(new Int32Array([1, 3, 3]));
      expect(new Float32Array(modelArtifacts.weightData.slice(12, 20)))
          .toEqual(new Float32Array([-7, -4]));
    });

    it('Missing modelTopology and weightsManifest leads to error',
       async () => {
         setupFakeWeightFiles(
             {
               'path1/model.json':
                   {data: JSON.stringify({}), contentType: 'application/json'}
             },
             requestInits);
         const handler = tf.io.http('path1/model.json');
         handler.load()
             .then(modelTopology1 => {
               fail(
                   'Loading from missing modelTopology and weightsManifest ' +
                   'succeeded unexpectedly.');
             })
             .catch(err => {
               expect(err.message)
                   .toMatch(/contains neither model topology or manifest/);
             });
       });

    it('with fetch rejection leads to error', async () => {
      setupFakeWeightFiles(
          {
            'path1/model.json':
                {data: JSON.stringify({}), contentType: 'text/html'}
          },
          requestInits);
      const handler = tf.io.http('path2/model.json');
      try {
        const data = await handler.load();
        expect(data).toBeDefined();
        fail('Loading with fetch rejection succeeded unexpectedly.');
      } catch (err) {
        // This error is mocked in beforeEach
        expect(err).toEqual('path not found');
      }
    });
    it('Provide WeightFileTranslateFunc', async () => {
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
            shape: [2],
            dtype: 'float32',
          }
        ]
      }];
      const floatData = new Float32Array([1, 3, 3, 7, 4]);
      setupFakeWeightFiles(
          {
            './model.json': {
              data: JSON.stringify({
                modelTopology: modelTopology1,
                weightsManifest: weightManifest1
              }),
              contentType: 'application/json'
            },
            'auth_weightfile0':
                {data: floatData, contentType: 'application/octet-stream'},
          },
          requestInits);
      async function prefixWeightUrlConverter(weightFile: string):
          Promise<string> {
        // Add 'auth_' prefix to the weight file url.
        return new Promise(
            resolve => setTimeout(resolve, 1, 'auth_' + weightFile));
      }

      const handler = tf.io.http('./model.json', {
        requestInit: {headers: {'header_key_1': 'header_value_1'}},
        weightUrlConverter: prefixWeightUrlConverter
      });
      const modelArtifacts = await handler.load();
      expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
      expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
      expect(new Float32Array(modelArtifacts.weightData)).toEqual(floatData);
      expect(Object.keys(requestInits).length).toEqual(2);
      expect(Object.keys(requestInits).length).toEqual(2);
      expect(requestInits['./model.json'].headers['header_key_1'])
          .toEqual('header_value_1');
      expect(requestInits['auth_weightfile0'].headers['header_key_1'])
          .toEqual('header_value_1');

      expect(fetchSpy.calls.mostRecent().object).toEqual(window);
    });
  });

  it('Overriding BrowserHTTPRequest fetchFunc', async () => {
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
          shape: [2],
          dtype: 'float32',
        }
      ]
    }];
    const floatData = new Float32Array([1, 3, 3, 7, 4]);

    const fetchInputs: RequestInfo[] = [];
    const fetchInits: RequestInit[] = [];
    async function customFetch(
        input: RequestInfo, init?: RequestInit): Promise<Response> {
      fetchInputs.push(input);
      fetchInits.push(init);

      if (input === './model.json') {
        return new Response(
            JSON.stringify({
              modelTopology: modelTopology1,
              weightsManifest: weightManifest1
            }),
            {status: 200, headers: {'content-type': 'application/json'}});
      } else if (input === './weightfile0') {
        return new Response(floatData, {
          status: 200,
          headers: {'content-type': 'application/octet-stream'}
        });
      } else {
        return new Response(null, {status: 404});
      }
    }

    const handler = tf.io.http(
        './model.json',
        {requestInit: {credentials: 'include'}, fetchFunc: customFetch});
    const modelArtifacts = await handler.load();
    expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
    expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
    expect(new Float32Array(modelArtifacts.weightData)).toEqual(floatData);

    expect(fetchInputs).toEqual(['./model.json', './weightfile0']);
    expect(fetchInits.length).toEqual(2);
    expect(fetchInits[0].credentials).toEqual('include');
    expect(fetchInits[1].credentials).toEqual('include');
  });
});
