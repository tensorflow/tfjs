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

import * as fm from './frozen_model';

const HOST = 'http://example.org';
const MODEL_URL = `${HOST}/model.pb`;
const WEIGHT_MANIFEST_URL = `${HOST}/weights_manifest.json`;
const RELATIVE_MODEL_URL = '/path/model.pb';
const RELATIVE_WEIGHT_MANIFEST_URL = '/path/weights_manifest.json';
let model: fm.FrozenModel;
const bias = tfc.tensor1d([1], 'int32');

const weightsManifest: tfc.io.WeightsManifestEntry[] =
    [{'name': 'Const', 'dtype': 'int32', 'shape': [1]}];

const SIMPLE_MODEL: tensorflow.IGraphDef = {
  node: [
    {
      name: 'Input',
      op: 'Placeholder',
      attr: {
        dtype: {
          type: tensorflow.DataType.DT_INT32,
        },
        shape: {shape: {dim: [{size: -1}, {size: 1}]}}
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
    {name: 'Add1', op: 'Add', input: ['Input', 'Const'], attr: {}},
    {name: 'Add', op: 'Add', input: ['Add1', 'Const'], attr: {}}
  ],
  versions: {producer: 1.0, minConsumer: 3}
};

const CONTROL_FLOW_MODEL: tensorflow.IGraphDef = {
  node: [
    {
      name: 'Input',
      op: 'Placeholder',
      attr: {
        dtype: {
          type: tensorflow.DataType.DT_INT32,
        },
        shape: {shape: {dim: [{size: -1}, {size: 1}]}}
      }
    },
    {name: 'Enter', op: 'Enter', attr: {}},
  ],
  versions: {producer: 1.0, minConsumer: 3}
};

const DYNAMIC_SHAPE_MODEL: tensorflow.IGraphDef = {
  node: [
    {
      name: 'Input',
      op: 'Placeholder',
      attr: {
        dtype: {
          type: tensorflow.DataType.DT_INT32,
        },
        shape: {shape: {dim: [{size: -1}, {size: 1}]}}
      }
    },
    {name: 'Where', op: 'Where', attr: {}}
  ],
  versions: {producer: 1.0, minConsumer: 3}
};
const HTTP_MODEL_LOADER = {
  load: async () => {
    return {
      modelTopology: new Uint8Array([1, 2, 3]),
      weightSpecs: weightsManifest,
      weightData: bias.dataSync()
    };
  }
};

describe('Model', () => {
  beforeEach(() => {
    model = new fm.FrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL);
    spyOn(tfc.io, 'getLoadHandlers').and.returnValue([HTTP_MODEL_LOADER]);
    spyOn(tfc.io, 'browserHTTPRequest').and.returnValue(HTTP_MODEL_LOADER);
  });
  afterEach(() => {});

  describe('simple model', () => {
    beforeEach(() => {
      spyOn(tensorflow.GraphDef, 'decode').and.returnValue(SIMPLE_MODEL);
    });

    it('load', async () => {
      const loaded = await model.load();
      expect(loaded).toBe(true);
    });

    describe('predict', () => {
      it('should generate the output for single tensor', async () => {
        await model.load();
        const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
        const output = model.predict(input);
        expect((output as tfc.Tensor).dataSync()[0]).toEqual(3);
      });

      it('should generate the output for tensor array', async () => {
        await model.load();
        const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
        const output = model.predict([input]);
        expect((output as tfc.Tensor).dataSync()[0]).toEqual(3);
      });

      it('should generate the output for tensor map', async () => {
        await model.load();
        const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
        const output = model.predict({'Input': input});
        expect((output as tfc.Tensor).dataSync()[0]).toEqual(3);
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

      it('should not allow feed intermediate node', async () => {
        await model.load();
        const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
        expect(() => model.predict({'Add1': input})).toThrow();
      });
    });

    describe('execute', () => {
      it('should generate the default output', async () => {
        await model.load();
        const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
        const output = model.execute({'Input': input});
        expect((output as tfc.Tensor).dataSync()[0]).toEqual(3);
      });
      it('should generate the output array', async () => {
        await model.load();
        const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
        const output = model.execute({'Input': input}, ['Add', 'Const']);
        expect(Array.isArray(output)).toBeTruthy();
        expect((output as tfc.Tensor[])[0].dataSync()[0]).toEqual(3);
        expect((output as tfc.Tensor[])[1].dataSync()[0]).toEqual(1);
      });
      it('should throw exception if inputs shapes do not match', () => {
        const input = tfc.tensor2d([1, 1], [1, 2], 'int32');
        expect(() => model.execute([input])).toThrow();
      });

      it('should throw exception if inputs dtype does not match graph', () => {
        const input = tfc.tensor2d([1, 1], [2, 1], 'float32');
        expect(() => model.predict([input])).toThrow();
      });

      it('should throw error if input size mismatch', async () => {
        await model.load();
        const input = tfc.tensor2d([1, 1], [2, 1], 'int32');

        expect(() => model.execute([input, input])).toThrow();
      });

      it('should allow feed intermediate node', async () => {
        await model.load();
        const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
        const output = model.execute({'Add1': input}) as tfc.Tensor;
        tfc.test_util.expectArraysClose(output, [2, 2]);
      });
    });

    describe('dispose', () => {
      it('should dispose the weights', async () => {
        const numOfTensors = tfc.memory().numTensors;
        model = new fm.FrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL);

        await model.load();
        model.dispose();

        expect(tfc.memory().numTensors).toEqual(numOfTensors);
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
        model = new fm.FrozenModel(
            RELATIVE_MODEL_URL, RELATIVE_WEIGHT_MANIFEST_URL);
      });

      it('load', async () => {
        const loaded = await model.load();
        expect(loaded).toBe(true);
      });
    });

    it('should loadFrozenModel', async () => {
      const model = await fm.loadFrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL);
      expect(model).not.toBeUndefined();
    });

    it('should loadFrozenModel with request options', async () => {
      const model = await fm.loadFrozenModel(
          MODEL_URL, WEIGHT_MANIFEST_URL, {credentials: 'include'});
      expect(tfc.io.browserHTTPRequest)
          .toHaveBeenCalledWith(
              [MODEL_URL, WEIGHT_MANIFEST_URL], {credentials: 'include'}, null,
              null, undefined);
      expect(model).not.toBeUndefined();
    });

    it('should call loadFrozenModel for loadTfHubModule', async () => {
      const url = `${HOST}/model/1`;
      const model = await fm.loadTfHubModule(url);
      expect(model).toBeDefined();
    });

    describe('InferenceModel interface', () => {
      it('should expose inputs', async () => {
        await model.load();
        expect(model.inputs).toEqual([
          {name: 'Input', shape: [-1, 1], dtype: 'int32'}
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

  describe('control flow model', () => {
    beforeEach(() => {
      spyOn(tensorflow.GraphDef, 'decode').and.returnValue(CONTROL_FLOW_MODEL);
    });

    it('should throw error if call predict directly', async () => {
      await model.load();
      const input = tfc.tensor2d([1, 1], [2, 1], 'int32');

      expect(() => model.predict([input])).toThrow();
    });

    it('should throw error if call execute directly', async () => {
      await model.load();
      const input = tfc.tensor2d([1, 1], [2, 1], 'int32');

      expect(() => model.predict([input])).toThrow();
    });

    it('should be success if call executeAsync', async () => {
      await model.load();
      const input = tfc.tensor2d([1, 1], [2, 1], 'int32');

      expect(() => model.executeAsync([input])).not.toThrow();
    });

    it('should allow feed intermediate node with executeAsync', async () => {
      await model.load();
      const input = tfc.tensor2d([1, 1], [2, 1], 'int32');

      expect(() => model.executeAsync({Enter: input})).not.toThrow();
    });
  });

  describe('dynamic shape model', () => {
    beforeEach(() => {
      spyOn(tensorflow.GraphDef, 'decode').and.returnValue(DYNAMIC_SHAPE_MODEL);
    });

    it('should throw error if call predict directly', async () => {
      await model.load();
      const input = tfc.tensor2d([1, 1], [2, 1], 'int32');

      expect(() => model.predict([input])).toThrow();
    });

    it('should throw error if call execute directly', async () => {
      await model.load();
      const input = tfc.tensor2d([1, 1], [2, 1], 'int32');

      expect(() => model.execute([input])).toThrow();
    });

    it('should be success if call executeAsync', async () => {
      await model.load();
      const input = tfc.tensor2d([1, 1], [2, 1], 'int32');

      expect(() => model.executeAsync([input])).not.toThrow();
    });

    it('should allow feed intermediate node with executeAsync', async () => {
      await model.load();
      const input = tfc.tensor2d([1, 1], [2, 1], 'int32');

      expect(() => model.executeAsync({Where: input})).not.toThrow();
    });
  });
});
