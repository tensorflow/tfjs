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

import * as data from '../data/index';

import {FrozenModel, loadFrozenModel} from './index';

const MODEL_URL = 'http://example.org/model.pb';
const WEIGHT_MANIFEST_URL = 'http://example.org/weights_manifest.json';
const RELATIVE_MODEL_URL = '/path/model.pb';
const RELATIVE_WEIGHT_MANIFEST_URL = '/path/weights_manifest.json';
let model: FrozenModel;
const bias = tfc.tensor1d([1], 'int32');
const WEIGHT_MAP = {
  'Const': bias
};
const SIMPLE_MODEL: data.tensorflow.IGraphDef = {
  node: [
    {
      name: 'Input',
      op: 'Placeholder',
      attr: {
        dtype: {
          type: data.tensorflow.DataType.DT_INT32,
        },
        shape: {shape: {dim: [{size: 1}]}}
      }
    },
    {
      name: 'Const',
      op: 'Const',
      attr: {
        dtype: {type: data.tensorflow.DataType.DT_INT32},
        value: {
          tensor: {
            dtype: data.tensorflow.DataType.DT_INT32,
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
    spyOn(data.tensorflow.GraphDef, 'decode').and.returnValue(SIMPLE_MODEL);
    const weightPromise = new Promise((resolve => resolve(WEIGHT_MAP)));
    spyOn(tfc, 'loadWeights').and.returnValue(weightPromise);
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

  describe('eval', () => {
    it('should generate the output', async () => {
      await model.load();
      const input = tfc.tensor1d([1], 'int32');
      const output = model.execute({'Input': input}, 'Add');
      expect((output as tfc.Tensor).dataSync()[0]).toEqual(2);
    });
  });

  describe('dispose', async () => {
    it('should dispose the weights', async () => {
      model = new FrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL);
      spyOn(bias, 'dispose');

      await model.load();
      model.dispose();

      expect(bias.dispose).toHaveBeenCalled();
    });
  });

  describe('getVersion', async () => {
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

  describe('loadFrozenModel', async () => {
    const model = await loadFrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL);
    expect(model).not.toBeUndefined();
  });
});
