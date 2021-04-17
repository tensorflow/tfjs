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
import {BROWSER_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose, expectArraysEqual} from '../test_util';
import {WeightsManifestConfig} from './types';

describeWithFlags('loadWeights', BROWSER_ENVS, () => {
  const setupFakeWeightFiles = (fileBufferMap: {
    [filename: string]: Float32Array|Int32Array|ArrayBuffer|Uint8Array|
    Uint16Array
  }) => {
    spyOn(tf.env().platform, 'fetch').and.callFake((path: string) => {
      return new Response(
          fileBufferMap[path],
          {headers: {'Content-type': 'application/octet-stream'}});
    });
  };

  it('1 group, 1 weight, 1 requested weight', async () => {
    setupFakeWeightFiles({'./weightfile0': new Float32Array([1, 2, 3])});

    const manifest: WeightsManifestConfig = [{
      'paths': ['weightfile0'],
      'weights': [{'name': 'weight0', 'dtype': 'float32', 'shape': [3]}]
    }];

    const weightsNamesToFetch = ['weight0'];
    const weights =
        await tf.io.loadWeights(manifest, './', weightsNamesToFetch);
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(1);

    const weightNames = Object.keys(weights);
    expect(weightNames.length).toEqual(weightsNamesToFetch.length);

    const weight0 = weights['weight0'];
    expectArraysClose(await weight0.data(), [1, 2, 3]);
    expect(weight0.shape).toEqual([3]);
    expect(weight0.dtype).toEqual('float32');
  });

  it('1 group, 2 weights, fetch 1st weight', async () => {
    setupFakeWeightFiles({'./weightfile0': new Float32Array([1, 2, 3, 4, 5])});

    const manifest: WeightsManifestConfig = [{
      'paths': ['weightfile0'],
      'weights': [
        {'name': 'weight0', 'dtype': 'float32', 'shape': [2]},
        {'name': 'weight1', 'dtype': 'float32', 'shape': [3]}
      ]
    }];

    // Load the first weight.
    const weights = await tf.io.loadWeights(manifest, './', ['weight0']);
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(1);

    const weightNames = Object.keys(weights);
    expect(weightNames.length).toEqual(1);

    const weight0 = weights['weight0'];
    expectArraysClose(await weight0.data(), [1, 2]);
    expect(weight0.shape).toEqual([2]);
    expect(weight0.dtype).toEqual('float32');
  });

  it('1 group, 2 weights, fetch 2nd weight', async () => {
    setupFakeWeightFiles({'./weightfile0': new Float32Array([1, 2, 3, 4, 5])});

    const manifest: WeightsManifestConfig = [{
      'paths': ['weightfile0'],
      'weights': [
        {'name': 'weight0', 'dtype': 'float32', 'shape': [2]},
        {'name': 'weight1', 'dtype': 'float32', 'shape': [3]}
      ]
    }];

    // Load the second weight.
    const weights = await tf.io.loadWeights(manifest, './', ['weight1']);
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(1);

    const weightNames = Object.keys(weights);
    expect(weightNames.length).toEqual(1);

    const weight1 = weights['weight1'];
    expectArraysClose(await weight1.data(), [3, 4, 5]);
    expect(weight1.shape).toEqual([3]);
    expect(weight1.dtype).toEqual('float32');
  });

  it('1 group, 2 weights, fetch all weights', async () => {
    setupFakeWeightFiles({'./weightfile0': new Float32Array([1, 2, 3, 4, 5])});

    const manifest: WeightsManifestConfig = [{
      'paths': ['weightfile0'],
      'weights': [
        {'name': 'weight0', 'dtype': 'float32', 'shape': [2]},
        {'name': 'weight1', 'dtype': 'float32', 'shape': [3]}
      ]
    }];

    // Load all weights.
    const weights =
        await tf.io.loadWeights(manifest, './', ['weight0', 'weight1']);
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(1);

    const weightNames = Object.keys(weights);
    expect(weightNames.length).toEqual(2);

    const weight0 = weights['weight0'];
    expectArraysClose(await weight0.data(), [1, 2]);
    expect(weight0.shape).toEqual([2]);
    expect(weight0.dtype).toEqual('float32');

    const weight1 = weights['weight1'];
    expectArraysClose(await weight1.data(), [3, 4, 5]);
    expect(weight1.shape).toEqual([3]);
    expect(weight1.dtype).toEqual('float32');
  });

  it('1 group, multiple weights, different dtypes', async () => {
    const buffer = new ArrayBuffer(5 * 4 + 1);
    const view = new DataView(buffer);
    view.setInt32(0, 1, true);
    view.setInt32(4, 2, true);
    view.setUint8(8, 1);
    view.setFloat32(9, 3., true);
    view.setFloat32(13, 4., true);
    view.setFloat32(17, 5., true);
    setupFakeWeightFiles({'./weightfile0': buffer});

    const manifest: WeightsManifestConfig = [{
      'paths': ['weightfile0'],
      'weights': [
        {'name': 'weight0', 'dtype': 'int32', 'shape': [2]},
        {'name': 'weight1', 'dtype': 'bool', 'shape': []},
        {'name': 'weight2', 'dtype': 'float32', 'shape': [3]},
      ]
    }];

    // Load all weights.
    const weights = await tf.io.loadWeights(
        manifest, './', ['weight0', 'weight1', 'weight2']);
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(1);

    const weightNames = Object.keys(weights);
    expect(weightNames.length).toEqual(3);

    const weight0 = weights['weight0'];
    expectArraysClose(await weight0.data(), [1, 2]);
    expect(weight0.shape).toEqual([2]);
    expect(weight0.dtype).toEqual('int32');

    const weight1 = weights['weight1'];
    expectArraysClose(await weight1.data(), [1]);
    expect(weight1.shape).toEqual([]);
    expect(weight1.dtype).toEqual('bool');

    const weight2 = weights['weight2'];
    expectArraysClose(await weight2.data(), [3, 4, 5]);
    expect(weight2.shape).toEqual([3]);
    expect(weight2.dtype).toEqual('float32');
  });

  it('1 group, sharded 1 weight across multiple files', async () => {
    const shard0 = new Float32Array([1, 2, 3, 4, 5]);
    const shard1 = new Float32Array([1.1, 2.2]);
    const shard2 = new Float32Array([10, 20, 30]);

    setupFakeWeightFiles({
      './weightfile0': shard0,
      './weightsfile1': shard1,
      './weightsfile2': shard2
    });

    const manifest: WeightsManifestConfig = [{
      'paths': ['weightfile0', 'weightsfile1', 'weightsfile2'],
      'weights': [{'name': 'weight0', 'dtype': 'float32', 'shape': [5, 2]}]
    }];

    const weights = await tf.io.loadWeights(manifest, './', ['weight0']);
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(3);

    const weightNames = Object.keys(weights);
    expect(weightNames.length).toEqual(1);

    const weight0 = weights['weight0'];
    expectArraysClose(
        await weight0.data(), [1, 2, 3, 4, 5, 1.1, 2.2, 10, 20, 30]);
    expect(weight0.shape).toEqual([5, 2]);
    expect(weight0.dtype).toEqual('float32');
  });

  it('1 group, sharded 2 weights across multiple files', async () => {
    const shard0 = new Int32Array([1, 2, 3, 4, 5]);

    // shard1 contains part of the first weight and part of the second.
    const shard1 = new ArrayBuffer(5 * 4);
    const intBuffer = new Int32Array(shard1, 0, 2);
    intBuffer.set([10, 20]);
    const floatBuffer = new Float32Array(shard1, intBuffer.byteLength, 3);
    floatBuffer.set([3.0, 4.0, 5.0]);

    const shard2 = new Float32Array([10, 20, 30]);

    setupFakeWeightFiles({
      './weightfile0': shard0,
      './weightsfile1': shard1,
      './weightsfile2': shard2
    });

    const manifest: WeightsManifestConfig = [{
      'paths': ['weightfile0', 'weightsfile1', 'weightsfile2'],
      'weights': [
        {'name': 'weight0', 'dtype': 'int32', 'shape': [7, 1]},
        {'name': 'weight1', 'dtype': 'float32', 'shape': [3, 2]}
      ]
    }];

    const weights =
        await tf.io.loadWeights(manifest, './', ['weight0', 'weight1']);
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(3);

    const weightNames = Object.keys(weights);
    expect(weightNames.length).toEqual(2);

    const weight0 = weights['weight0'];
    expectArraysClose(await weight0.data(), [1, 2, 3, 4, 5, 10, 20]);
    expect(weight0.shape).toEqual([7, 1]);
    expect(weight0.dtype).toEqual('int32');

    const weight1 = weights['weight1'];
    expectArraysClose(await weight1.data(), [3.0, 4.0, 5.0, 10, 20, 30]);
    expect(weight1.shape).toEqual([3, 2]);
    expect(weight1.dtype).toEqual('float32');
  });

  it('2 group, 4 weights, fetches one group', async () => {
    setupFakeWeightFiles({
      './weightfile0': new Float32Array([1, 2, 3, 4, 5]),
      './weightfile1': new Float32Array([6, 7, 8, 9])
    });

    const manifest: WeightsManifestConfig = [
      {
        'paths': ['weightfile0'],
        'weights': [
          {'name': 'weight0', 'dtype': 'float32', 'shape': [2]},
          {'name': 'weight1', 'dtype': 'float32', 'shape': [3]}
        ]
      },
      {
        'paths': ['weightfile1'],
        'weights': [
          {'name': 'weight2', 'dtype': 'float32', 'shape': [3, 1]},
          {'name': 'weight3', 'dtype': 'float32', 'shape': []}
        ]
      }
    ];

    const weights =
        await tf.io.loadWeights(manifest, './', ['weight0', 'weight1']);
    // Only the first group should be fetched.
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(1);

    const weightNames = Object.keys(weights);
    expect(weightNames.length).toEqual(2);

    const weight0 = weights['weight0'];
    expectArraysClose(await weight0.data(), [1, 2]);
    expect(weight0.shape).toEqual([2]);
    expect(weight0.dtype).toEqual('float32');

    const weight1 = weights['weight1'];
    expectArraysClose(await weight1.data(), [3, 4, 5]);
    expect(weight1.shape).toEqual([3]);
    expect(weight1.dtype).toEqual('float32');
  });

  it('2 group, 4 weights, one weight from each group', async () => {
    setupFakeWeightFiles({
      './weightfile0': new Float32Array([1, 2, 3, 4, 5]),
      './weightfile1': new Float32Array([6, 7, 8, 9])
    });

    const manifest: WeightsManifestConfig = [
      {
        'paths': ['weightfile0'],
        'weights': [
          {'name': 'weight0', 'dtype': 'float32', 'shape': [2]},
          {'name': 'weight1', 'dtype': 'float32', 'shape': [3]}
        ]
      },
      {
        'paths': ['weightfile1'],
        'weights': [
          {'name': 'weight2', 'dtype': 'float32', 'shape': [3, 1]},
          {'name': 'weight3', 'dtype': 'float32', 'shape': []}
        ]
      }
    ];

    const weights =
        await tf.io.loadWeights(manifest, './', ['weight0', 'weight2']);
    // Both groups need to be fetched.
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(2);

    const weightNames = Object.keys(weights);
    expect(weightNames.length).toEqual(2);

    const weight0 = weights['weight0'];
    expectArraysClose(await weight0.data(), [1, 2]);
    expect(weight0.shape).toEqual([2]);
    expect(weight0.dtype).toEqual('float32');

    const weight2 = weights['weight2'];
    expectArraysClose(await weight2.data(), [6, 7, 8]);
    expect(weight2.shape).toEqual([3, 1]);
    expect(weight2.dtype).toEqual('float32');
  });

  it('2 group, 4 weights, dont specify weights fetchs all', async () => {
    setupFakeWeightFiles({
      './weightfile0': new Float32Array([1, 2, 3, 4, 5]),
      './weightfile1': new Float32Array([6, 7, 8, 9])
    });

    const manifest: WeightsManifestConfig = [
      {
        'paths': ['weightfile0'],
        'weights': [
          {'name': 'weight0', 'dtype': 'float32', 'shape': [2]},
          {'name': 'weight1', 'dtype': 'float32', 'shape': [3]}
        ]
      },
      {
        'paths': ['weightfile1'],
        'weights': [
          {'name': 'weight2', 'dtype': 'float32', 'shape': [3, 1]},
          {'name': 'weight3', 'dtype': 'float32', 'shape': []}
        ]
      }
    ];

    // Don't pass a third argument to loadWeights to load all weights.
    const weights = await tf.io.loadWeights(manifest, './');
    // Both groups need to be fetched.
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(2);

    const weightNames = Object.keys(weights);
    expect(weightNames.length).toEqual(4);

    const weight0 = weights['weight0'];
    expectArraysClose(await weight0.data(), [1, 2]);
    expect(weight0.shape).toEqual([2]);
    expect(weight0.dtype).toEqual('float32');

    const weight1 = weights['weight1'];
    expectArraysClose(await weight1.data(), [3, 4, 5]);
    expect(weight1.shape).toEqual([3]);
    expect(weight1.dtype).toEqual('float32');

    const weight2 = weights['weight2'];
    expectArraysClose(await weight2.data(), [6, 7, 8]);
    expect(weight2.shape).toEqual([3, 1]);
    expect(weight2.dtype).toEqual('float32');

    const weight3 = weights['weight3'];
    expectArraysClose(await weight3.data(), [9]);
    expect(weight3.shape).toEqual([]);
    expect(weight3.dtype).toEqual('float32');
  });

  it('throws if requested weight not found', async () => {
    setupFakeWeightFiles({'./weightfile0': new Float32Array([1, 2, 3])});

    const manifest: WeightsManifestConfig = [{
      'paths': ['weightfile0'],
      'weights': [{'name': 'weight0', 'dtype': 'float32', 'shape': [3]}]
    }];

    const weightsNamesToFetch = ['doesntexist'];
    try {
      await tf.io.loadWeights(manifest, './', weightsNamesToFetch);
      fail();
    } catch (e) {
      expect(e.message).toContain('Could not find weights');
    }
  });

  it('throws if requested weight has unknown dtype', async () => {
    setupFakeWeightFiles({'./weightfile0': new Float32Array([1, 2, 3])});

    const manifest: WeightsManifestConfig = [{
      'paths': ['weightfile0'],
      'weights': [{
        'name': 'weight0',
        // tslint:disable-next-line:no-any
        'dtype': 'null' as any,
        'shape': [3]
      }]
    }];

    const weightsNamesToFetch = ['weight0'];
    try {
      await tf.io.loadWeights(manifest, './', weightsNamesToFetch);
      fail();
    } catch (e) {
      expect(e.message).toContain('Unsupported dtype');
    }
  });

  it('should use request option', async () => {
    setupFakeWeightFiles({'./weightfile0': new Float32Array([1, 2, 3])});

    const manifest: WeightsManifestConfig = [{
      'paths': ['weightfile0'],
      'weights': [{'name': 'weight0', 'dtype': 'float32', 'shape': [3]}]
    }];

    const weightsNamesToFetch = ['weight0'];
    await tf.io.loadWeights(
        manifest, './', weightsNamesToFetch, {credentials: 'include'});
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(1);
    expect(tf.env().platform.fetch)
        .toHaveBeenCalledWith(
            './weightfile0', {credentials: 'include'}, {isBinary: true});
  });

  const quantizationTest = async (quantizationDtype: 'uint8'|'uint16') => {
    const arrayType = quantizationDtype === 'uint8' ? Uint8Array : Uint16Array;
    setupFakeWeightFiles(
        {'./weightfile0': new arrayType([0, 48, 255, 0, 48, 255])});

    const manifest: WeightsManifestConfig = [{
      'paths': ['weightfile0'],
      'weights': [
        {
          'name': 'weight0',
          'dtype': 'float32',
          'shape': [3],
          'quantization': {'min': -1, 'scale': 0.1, 'dtype': quantizationDtype}
        },
        {
          'name': 'weight1',
          'dtype': 'int32',
          'shape': [3],
          'quantization': {'min': -1, 'scale': 0.1, 'dtype': quantizationDtype}
        }
      ]
    }];

    const weightsNamesToFetch = ['weight0', 'weight1'];
    const weights =
        await tf.io.loadWeights(manifest, './', weightsNamesToFetch);
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(1);

    const weightNames = Object.keys(weights);
    expect(weightNames.length).toEqual(weightsNamesToFetch.length);

    const weight0 = weights['weight0'];
    expectArraysClose(await weight0.data(), [-1, 3.8, 24.5]);
    expect(weight0.shape).toEqual([3]);
    expect(weight0.dtype).toEqual('float32');

    const weight1 = weights['weight1'];
    expectArraysEqual(await weight1.data(), [-1, 4, 25]);
    expect(weight1.shape).toEqual([3]);
    expect(weight1.dtype).toEqual('int32');
  };

  it('quantized weights (uint8)', async () => {
    await quantizationTest('uint8');
  });

  it('quantized weights (uint16)', async () => {
    await quantizationTest('uint16');
  });

  it('2 groups, 1 quantized, 1 unquantized', async () => {
    setupFakeWeightFiles({
      './weightfile0': new Uint8Array([0, 48, 255, 0, 48, 255]),
      './weightfile1': new Float32Array([6, 7, 8, 9])
    });

    const manifest: WeightsManifestConfig = [
      {
        'paths': ['weightfile0'],
        'weights': [
          {
            'name': 'weight0',
            'dtype': 'float32',
            'shape': [3],
            'quantization': {'min': -1, 'scale': 0.1, 'dtype': 'uint8'}
          },
          {
            'name': 'weight1',
            'dtype': 'int32',
            'shape': [3],
            'quantization': {'min': -1, 'scale': 0.1, 'dtype': 'uint8'}
          }
        ]
      },
      {
        'paths': ['weightfile1'],
        'weights': [
          {'name': 'weight2', 'dtype': 'float32', 'shape': [3, 1]},
          {'name': 'weight3', 'dtype': 'float32', 'shape': []}
        ]
      }
    ];

    const weights =
        await tf.io.loadWeights(manifest, './', ['weight0', 'weight2']);
    // Both groups need to be fetched.
    expect((tf.env().platform.fetch as jasmine.Spy).calls.count()).toBe(2);

    const weightNames = Object.keys(weights);
    expect(weightNames.length).toEqual(2);

    const weight0 = weights['weight0'];
    expectArraysClose(await weight0.data(), [-1, 3.8, 24.5]);
    expect(weight0.shape).toEqual([3]);
    expect(weight0.dtype).toEqual('float32');

    const weight2 = weights['weight2'];
    expectArraysClose(await weight2.data(), [6, 7, 8]);
    expect(weight2.shape).toEqual([3, 1]);
    expect(weight2.dtype).toEqual('float32');
  });
});
