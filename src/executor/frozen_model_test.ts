/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as tfc from '@tensorflow/tfjs-core';

import {tensorflow} from '../data/compiled_api';

import {FrozenModel, loadFrozenModel} from './frozen_model';

const MODEL_URL = 'http://example.org/model.pb';
const WEIGHT_MANIFEST_URL = 'http://example.org/weights_manifest.json';
const RELATIVE_MODEL_URL = '/path/model.pb';
const RELATIVE_WEIGHT_MANIFEST_URL = '/path/weights_manifest.json';
let model: FrozenModel;
const bias = tfc.tensor1d([1], 'int32');
const WEIGHT_MAP = {
  'Const': bias
};
const SIMPLE_MODEL: tensorflow.IGraphDef = {
  node: [
    {
      name: 'Input',
      op: 'Placeholder',
      attr: {
        dtype: {
          type: tensorflow.DataType.DT_INT32,
        },
        shape: {shape: {dim: [{size: 1}]}}
      }
    },
    {
      name: 'Const',
      op: 'Const',
      attr: {
        dtype: {type: tensorflow.DataType.DT_INT32},
        value: {
          tensor: {
            dtype: tensorflow.DataType.DT_INT32,
            tensorShape: {dim: [{size: 1}]},
          }
        },
        index: {i: 0},
        length: {i: 4}
      }
    },
    {name: 'Add', op: 'Add', input: ['Input', 'Const'], attr: {}}
  ],
  versions: {producer: 1.0, minConsumer: 3}
};

describe('Model', () => {
  beforeEach(() => {
    spyOn(tensorflow.GraphDef, 'decode').and.returnValue(SIMPLE_MODEL);
    const weightPromise = new Promise((resolve => resolve(WEIGHT_MAP)));
    spyOn(tfc.io, 'loadWeights').and.returnValue(weightPromise);
    model = new FrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL);
    spyOn(window, 'fetch')
        .and.callFake(
            () => new Promise(
                (resolve =>
                     resolve(new Response(JSON.stringify({json: 'ok!'}))))));
  });
  afterEach(() => {});

  it('load', async () => {
    const loaded = await model.load();
    expect(loaded).toBe(true);
  });

  describe('getPathPrefix', () => {
    it('should set pathPrefix (absolute path)', async () => {
      model = new FrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL);
      expect(model.getPathPrefix()).toEqual('http://example.org/');
    });

    it('should set pathPrefix (relative path)', async () => {
      model = new FrozenModel(RELATIVE_MODEL_URL, RELATIVE_WEIGHT_MANIFEST_URL);
      expect(model.getPathPrefix()).toEqual('/path/');
    });
  });
  describe('predict', () => {
    it('should generate the output for single tensor', async () => {
      await model.load();
      const input = tfc.tensor1d([1], 'int32');
      const output = model.predict(input);
      expect((output as tfc.Tensor).dataSync()[0]).toEqual(2);
    });

    it('should generate the output for tensor array', async () => {
      await model.load();
      const input = tfc.tensor1d([1], 'int32');
      const output = model.predict([input]);
      expect((output as tfc.Tensor).dataSync()[0]).toEqual(2);
    });

    it('should generate the output for tensor map', async () => {
      await model.load();
      const input = tfc.tensor1d([1], 'int32');
      const output = model.predict({'Input': input});
      expect((output as tfc.Tensor).dataSync()[0]).toEqual(2);
    });

    it('should throw error if input size mismatch', async () => {
      await model.load();
      const input = tfc.tensor1d([1], 'int32');
      expect(() => model.predict([input, input])).toThrow();
    });

    it('should throw exception if inputs shapes do not match', () => {
      const input = tfc.tensor2d([1, 1], [1, 2], 'int32');
      expect(() => model.predict([input])).toThrow();
    });

    it('should throw exception if inputs dtype does not match graph', () => {
      const input = tfc.tensor1d([1], 'float32');
      expect(() => model.predict([input])).toThrow();
    });
  });

  describe('execute', () => {
    it('should generate the default output', async () => {
      await model.load();
      const input = tfc.tensor1d([1], 'int32');
      const output = model.execute({'Input': input});
      expect((output as tfc.Tensor).dataSync()[0]).toEqual(2);
    });
    it('should generate the output array', async () => {
      await model.load();
      const input = tfc.tensor1d([1], 'int32');
      const output = model.execute({'Input': input}, ['Add', 'Const']);
      expect(Array.isArray(output)).toBeTruthy();
      expect((output as tfc.Tensor[])[0].dataSync()[0]).toEqual(2);
      expect((output as tfc.Tensor[])[1].dataSync()[0]).toEqual(1);
    });
    it('should throw exception if inputs shapes do not match', () => {
      const input = tfc.tensor2d([1, 1], [1, 2], 'int32');
      expect(() => model.execute([input])).toThrow();
    });

    it('should throw exception if inputs dtype does not match graph', () => {
      const input = tfc.tensor1d([1], 'float32');
      expect(() => model.predict([input])).toThrow();
    });

    it('should throw error if input size mismatch', async () => {
      await model.load();
      const input = tfc.tensor1d([1], 'int32');
      expect(() => model.execute([input, input])).toThrow();
    });
  });

  describe('dispose', () => {
    it('should dispose the weights', async () => {
      model = new FrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL);
      spyOn(bias, 'dispose');

      await model.load();
      model.dispose();

      expect(bias.dispose).toHaveBeenCalled();
    });
  });

  describe('getVersion', () => {
    it('should return the version info from the tf model', async () => {
      await model.load();
      expect(model.modelVersion).toEqual('1.3');
    });
  });

  describe('relative path', () => {
    beforeEach(() => {
      model = new FrozenModel(RELATIVE_MODEL_URL, RELATIVE_WEIGHT_MANIFEST_URL);
    });

    it('load', async () => {
      const loaded = await model.load();
      expect(loaded).toBe(true);
    });
  });

  it('should loadFrozenModel', async () => {
    const model = await loadFrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL);
    expect(model).not.toBeUndefined();
  });

  it('should loadFrozenModel with request options', async () => {
    const model = await loadFrozenModel(
        MODEL_URL, WEIGHT_MANIFEST_URL, {credentials: 'include'});
    expect(window.fetch).toHaveBeenCalledWith(MODEL_URL, {
      credentials: 'include'
    });
    expect(window.fetch).toHaveBeenCalledWith(WEIGHT_MANIFEST_URL, {
      credentials: 'include'
    });
    expect(model).not.toBeUndefined();
  });

  describe('InferenceModel interface', () => {
    it('should expose inputs', async () => {
      await model.load();
      expect(model.inputs).toEqual([
        {name: 'Input', shape: [1], dtype: 'int32'}
      ]);
    });
    it('should expose outputs', async () => {
      await model.load();
      expect(model.outputs).toEqual([
        {name: 'Add', shape: undefined, dtype: undefined}
      ]);
    });
  });
});
