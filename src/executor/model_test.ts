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

import * as dl from 'deeplearn';
import * as data from '../data/index';
import {Model} from './index';

const MODEL_URL = 'model url';
const WEIGHT_URL = 'weight url';
let model: Model;
const WEIGHT_ARRAY = new Int32Array([1]);
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
    const graphPromise = new Promise((resolve => resolve(SIMPLE_MODEL)));
    spyOn(data, 'loadRemoteProtoFile').and.returnValue(graphPromise);
    const weightPromise = new Promise((resolve => resolve(WEIGHT_ARRAY)));
    spyOn(data, 'loadRemoteWeightFile').and.returnValue(weightPromise);
    model = new Model(MODEL_URL, WEIGHT_URL);
  });
  afterEach(() => {});

  describe('load', async () => {
    const loaded = await model.load();
    expect(loaded).toBe(true);
  });

  describe('predict', () => {
    it('should generate the output', async () => {
      await model.load();
      const input = dl.tensor1d([1], 'int32');
      const output = model.predict({'Input': input});
      expect(Object.keys(output)).toEqual(['Add']);
      expect(output['Add'].dataSync()[0]).toEqual(2);
    });
  });

  describe('dispose', async () => {
    it('should dispose the weights', async () => {
      const bias = dl.tensor1d([1], 'int32');
      spyOn(data, 'buildWeightMap').and.returnValue({'Const': bias});
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
});
