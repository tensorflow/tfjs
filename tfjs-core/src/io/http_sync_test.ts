/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import {BROWSER_ENVS, describeWithFlags} from '../jasmine_util';
import {Platform} from '../platforms/platform';
import {modelTopology1, trainingConfig1, TypedArrays} from './http_test';

let fetchSpy: jasmine.Spy;

export const fakeResponse =
    (body: string|TypedArrays|ArrayBuffer, contentType: string, path: string) =>
        ({
          ok: true,
          json() {
            return JSON.parse(body as string);
          },
          arrayBuffer() {
            const buf: ArrayBuffer = (body as TypedArrays).buffer ?
                (body as TypedArrays).buffer :
                body as ArrayBuffer;
            return buf;
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
       fetchSpy = spyOn(tf.env().platform, 'fetchSync')
                     .and.callFake((path: string, init: RequestInit) => {
                       if (fileBufferMap[path]) {
                         requestInits[path] = init;
                         return fakeResponse(
                             fileBufferMap[path].data,
                             fileBufferMap[path].contentType, path);
                       } else {
                         throw new Error('path not found');
                       }
                     });
    };

describeWithFlags('http-sync-load', BROWSER_ENVS, () => {
  describe('JSON model', () => {
    let requestInits: {[key: string]: {headers: {[key: string]: string}}};

    beforeEach(() => {
      requestInits = {};
    });

    it('1 group, 2 weights, 1 path', () => {
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

      const handler = tf.io.httpSync('./model.json');
      const modelArtifacts = handler.load();
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

    it('1 group, 2 weights, 1 path, with requestInit', () => {
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

      const handler = tf.io.httpSync(
          './model.json',
          {requestInit: {headers: {'header_key_1': 'header_value_1'}}});
      const modelArtifacts = handler.load();
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

    it('1 group, 2 weight, 2 paths', () => {
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

      const handler = tf.io.httpSync('./model.json');
      const modelArtifacts = handler.load();
      expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
      expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
      expect(new Float32Array(modelArtifacts.weightData))
          .toEqual(new Float32Array([1, 3, 3, 7, 4]));
    });

    it('2 groups, 2 weight, 2 paths', () => {
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

      const handler = tf.io.httpSync('./model.json');
      const modelArtifacts = handler.load();
      expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
      expect(modelArtifacts.weightSpecs)
          .toEqual(
              weightsManifest[0].weights.concat(weightsManifest[1].weights));
      expect(new Float32Array(modelArtifacts.weightData))
          .toEqual(new Float32Array([1, 3, 3, 7, 4]));
    });

    it('2 groups, 2 weight, 2 paths, Int32 and Uint8 Data', () => {
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

      const handler = tf.io.httpSync('path1/model.json');
      const modelArtifacts = handler.load();
      expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
      expect(modelArtifacts.weightSpecs)
          .toEqual(
              weightsManifest[0].weights.concat(weightsManifest[1].weights));
      expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
          .toEqual(new Int32Array([1, 3, 3]));
      expect(new Uint8Array(modelArtifacts.weightData.slice(12, 14)))
          .toEqual(new Uint8Array([7, 4]));
    });

    it('topology only', () => {
      setupFakeWeightFiles(
          {
            './model.json': {
              data: JSON.stringify({modelTopology: modelTopology1}),
              contentType: 'application/json'
            },
          },
          requestInits);

      const handler = tf.io.httpSync('./model.json');
      const modelArtifacts = handler.load();
      expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
      expect(modelArtifacts.weightSpecs).toBeUndefined();
      expect(modelArtifacts.weightData).toBeUndefined();
    });

    it('weights only', () => {
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

      const handler = tf.io.httpSync('path1/model.json');
      const modelArtifacts = handler.load();
      expect(modelArtifacts.modelTopology).toBeUndefined();
      expect(modelArtifacts.weightSpecs)
          .toEqual(
              weightsManifest[0].weights.concat(weightsManifest[1].weights));
      expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
          .toEqual(new Int32Array([1, 3, 3]));
      expect(new Float32Array(modelArtifacts.weightData.slice(12, 20)))
          .toEqual(new Float32Array([-7, -4]));
    });

    it('Missing modelTopology and weightsManifest leads to error', () => {
      setupFakeWeightFiles(
          {
            'path1/model.json':
                {data: JSON.stringify({}), contentType: 'application/json'}
          },
          requestInits);
      const handler = tf.io.httpSync('path1/model.json');
      try {
        handler.load();
        fail('Loading from missing modelTopology and weightsManifest ' +
          'succeeded unexpectedly.');
      }
      catch (err) {
        expect(err.message)
          .toMatch(/contains neither model topology or manifest/);
      }
    });

    it('with fetch error leads to error', () => {
      setupFakeWeightFiles(
          {
            'path1/model.json':
                {data: JSON.stringify({}), contentType: 'text/html'}
          },
          requestInits);
      const handler = tf.io.httpSync('path2/model.json');
      try {
        const data = handler.load();
        expect(data).toBeDefined();
        fail('Loading with fetch error succeeded unexpectedly.');
      } catch (err) {
        // This error is mocked in beforeEach
        expect((err as Error).message).toEqual('path not found');
      }
    });
    it('Provide WeightFileTranslateFunc', () => {
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
      function prefixWeightUrlConverter(weightFile: string): string {
        // Add 'auth_' prefix to the weight file url.
        return 'auth_' + weightFile;
      }

      const handler = tf.io.httpSync('./model.json', {
        requestInit: {headers: {'header_key_1': 'header_value_1'}},
        weightUrlConverter: prefixWeightUrlConverter
      });
      const modelArtifacts = handler.load();
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

  it('Overriding BrowserHTTPRequest fetchFunc', () => {
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
    function customFetch(input: RequestInfo, init?: RequestInit):
    ReturnType<Platform['fetchSync']> {
      fetchInputs.push(input);
      fetchInits.push(init);

      if (input === './model.json') {
        return {
          json() {
            return {
              modelTopology: modelTopology1,
              weightsManifest: weightManifest1,
              trainingConfig: trainingConfig1
            };
          },
          ok: true,
          arrayBuffer() {throw new Error('not supported');}
        };
      } else if (input === './weightfile0') {
        return {
          arrayBuffer() {
            return floatData;
          },
          json() {throw new Error('not supported');},
          ok: true,
        };
      } else {
        return {
          ok: false,
          json() {throw new Error('not supported');},
          arrayBuffer() {throw new Error('not supported');},
        };
      }
    }

    const handler = tf.io.httpSync(
        './model.json',
        {requestInit: {credentials: 'include'}, fetchFunc: customFetch});
    const modelArtifacts = handler.load();
    expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
    expect(modelArtifacts.trainingConfig).toEqual(trainingConfig1);
    expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
    expect(new Float32Array(modelArtifacts.weightData)).toEqual(floatData);

    expect(fetchInputs).toEqual(['./model.json', './weightfile0']);
    expect(fetchInits.length).toEqual(2);
    expect(fetchInits[0].credentials).toEqual('include');
    expect(fetchInits[1].credentials).toEqual('include');
  });
});
