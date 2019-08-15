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
import {scalar} from '@tensorflow/tfjs-core';

import * as tensorflow from '../data/compiled_api';
import {deregisterOp, registerOp} from '../operations/custom_op/register';
import {GraphNode} from '../operations/types';

import {GraphModel, loadGraphModel} from './graph_model';

const HOST = 'http://example.org';
const MODEL_URL = `${HOST}/model.json`;
const RELATIVE_MODEL_URL = '/path/model.pb';
let model: GraphModel;
const bias = tfc.tensor1d([1], 'int32');
const alpha = tfc.scalar(1, 'int32');

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

const PRELU_MODEL: tensorflow.IGraphDef = {
  node: [
    {
      name: 'Input',
      op: 'Placeholder',
      attr: {
        dtype: {
          type: tensorflow.DataType.DT_FLOAT,
        },
        shape: {shape: {dim: []}}
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
            tensorShape: {dim: []},
          }
        },
        index: {i: 0},
        length: {i: 4}
      }
    },
    {name: 'Relu', op: 'Relu', input: ['Input'], attr: {}},
    {name: 'Neg', op: 'Neg', input: ['Input'], attr: {}},
    {name: 'Relu2', op: 'Relu', input: ['Neg'], attr: {}},
    {name: 'Mul', op: 'Mul', input: ['Const', 'Relu2'], attr: {}},
    {name: 'Add', op: 'Add', input: ['Relu', 'Mul'], attr: {}}
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
    {name: 'Enter', op: 'Enter', attr: {}, input: ['Input']},
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
          type: tensorflow.DataType.DT_BOOL,
        },
        shape: {shape: {dim: [{size: -1}, {size: 1}]}}
      }
    },
    {name: 'Where', op: 'Where', attr: {}, input: ['Input']}
  ],
  versions: {producer: 1.0, minConsumer: 3}
};
const SIMPLE_HTTP_MODEL_LOADER = {
  load: async () => {
    return {
      modelTopology: SIMPLE_MODEL,
      weightSpecs: weightsManifest,
      weightData: bias.dataSync()
    };
  }
};
const PRELU_HTTP_MODEL_LOADER = {
  load: async () => {
    return {
      modelTopology: PRELU_MODEL,
      weightSpecs: weightsManifest,
      weightData: alpha.dataSync()
    };
  }
};
const CUSTOM_OP_MODEL: tensorflow.IGraphDef = {
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
    {name: 'CustomOp', op: 'CustomOp', input: ['Add1'], attr: {}}
  ],
  versions: {producer: 1.0, minConsumer: 3}
};

const CUSTOM_HTTP_MODEL_LOADER = {
  load: async () => {
    return {
      modelTopology: CUSTOM_OP_MODEL,
      weightSpecs: weightsManifest,
      weightData: bias.dataSync()
    };
  }
};

const CONTROL_FLOW_HTTP_MODEL_LOADER = {
  load: async () => {
    return {
      modelTopology: CONTROL_FLOW_MODEL,
      weightSpecs: weightsManifest,
      weightData: bias.dataSync()
    };
  }
};

describe('loadGraphModel', () => {
  it('Pass a custom io handler', async () => {
    const customLoader: tfc.io.IOHandler = {
      load: async () => {
        return {
          modelTopology: SIMPLE_MODEL,
          weightSpecs: weightsManifest,
          weightData: new Int32Array([5]).buffer,
        };
      }
    };
    const model = await loadGraphModel(customLoader);
    expect(model).toBeDefined();
    const bias = model.weights['Const'][0];
    expect(bias.dtype).toBe('int32');
    expect(bias.dataSync()).toEqual(new Int32Array([5]));
  });

  it('Expect an error when moderUrl is null', async () => {
    let errorMsg = 'no error';
    try {
      await loadGraphModel(null);
    } catch (err) {
      errorMsg = err.message;
    }
    expect(errorMsg).toMatch(/modelUrl in loadGraphModel\(\) cannot be null/);
  });
});

describe('Model', () => {
  beforeEach(() => {
    model = new GraphModel(MODEL_URL);
  });

  describe('custom model', () => {
    beforeEach(() => {
      spyOn(tfc.io, 'getLoadHandlers').and.returnValue([
        CUSTOM_HTTP_MODEL_LOADER
      ]);
      registerOp('CustomOp', (nodeValue: GraphNode) => {
        const x = nodeValue.inputs[0];
        return [tfc.add(x, scalar(1, 'int32'))];
      });
    });
    afterEach(() => deregisterOp('CustomOp'));
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
    });
  });

  describe('simple model', () => {
    beforeEach(() => {
      spyOn(tfc.io, 'getLoadHandlers').and.returnValue([
        SIMPLE_HTTP_MODEL_LOADER
      ]);
      spyOn(tfc.io, 'browserHTTPRequest')
          .and.returnValue(SIMPLE_HTTP_MODEL_LOADER);
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
        tfc.test_util.expectArraysClose(await output.data(), [2, 2]);
      });
    });

    describe('dispose', () => {
      it('should dispose the weights', async () => {
        const numOfTensors = tfc.memory().numTensors;
        model = new GraphModel(MODEL_URL);

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
        model = new GraphModel(RELATIVE_MODEL_URL);
      });

      it('load', async () => {
        const loaded = await model.load();
        expect(loaded).toBe(true);
      });
    });

    it('should loadGraphModel', async () => {
      const model = await loadGraphModel(MODEL_URL);
      expect(model).not.toBeUndefined();
    });

    it('should loadGraphModel with request options', async () => {
      const model = await loadGraphModel(
          MODEL_URL, {requestInit: {credentials: 'include'}});
      expect(tfc.io.browserHTTPRequest).toHaveBeenCalledWith(MODEL_URL, {
        requestInit: {credentials: 'include'}
      });
      expect(model).not.toBeUndefined();
    });

    it('should call loadGraphModel for TfHub Module', async () => {
      const url = `${HOST}/model/1`;
      const model = await loadGraphModel(url, {fromTFHub: true});
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

  describe('prelu op model', () => {
    beforeEach(() => {
      spyOn(tfc.io, 'getLoadHandlers').and.returnValue([
        PRELU_HTTP_MODEL_LOADER
      ]);
      spyOn(tfc.io, 'browserHTTPRequest')
          .and.returnValue(PRELU_HTTP_MODEL_LOADER);
    });

    it('fusePrelu should call model rewrite method', async () => {
      await model.load();
      const originalResult = model.predict(tfc.scalar(1)) as tfc.Tensor;

      model.fusePrelu();
      expect(Object.keys(model.weights)).toContain('Const_neg');
      expect(model.outputNodes[0]).toEqual('Add_Prelu');
      const result = model.predict(tfc.scalar(1)) as tfc.Tensor;

      tfc.test_util.expectArraysClose(
          result.dataSync(), originalResult.dataSync());
    });
  });

  describe('control flow model', () => {
    beforeEach(() => {
      spyOn(tfc.io, 'getLoadHandlers').and.returnValue([
        CONTROL_FLOW_HTTP_MODEL_LOADER
      ]);
      spyOn(tfc.io, 'browserHTTPRequest')
          .and.returnValue(CONTROL_FLOW_HTTP_MODEL_LOADER);
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
      const res = await model.executeAsync([input]);
      expect(res).not.toBeNull();
    });

    it('should allow feed intermediate node with executeAsync', async () => {
      await model.load();
      const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
      const res = await model.executeAsync({Enter: input});
      expect(res).not.toBeNull();
    });
  });
  const DYNAMIC_HTTP_MODEL_LOADER = {
    load: async () => {
      return {
        modelTopology: DYNAMIC_SHAPE_MODEL,
        weightSpecs: weightsManifest,
        weightData: bias.dataSync()
      };
    }
  };
  describe('dynamic shape model', () => {
    beforeEach(() => {
      spyOn(tfc.io, 'getLoadHandlers').and.returnValue([
        DYNAMIC_HTTP_MODEL_LOADER
      ]);
      spyOn(tfc.io, 'browserHTTPRequest')
          .and.returnValue(DYNAMIC_HTTP_MODEL_LOADER);
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
      const input = tfc.tensor2d([1, 1], [2, 1], 'bool');
      const res = await model.executeAsync([input]);
      expect(res).not.toBeNull();
    });

    it('should allow feed intermediate node with executeAsync', async () => {
      await model.load();
      const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
      const res = await model.executeAsync({Where: input});
      expect(res).not.toBeNull();
    });
  });
});

describe('Graph execution gives actionable errors', () => {
  it('executeAsync warns when there are no dynamic ops', async () => {
    const customLoader: tfc.io.IOHandler = {
      load: async () => ({
        modelTopology: SIMPLE_MODEL,
        weightSpecs: weightsManifest,
        weightData: new Int32Array([5]).buffer,
      })
    };
    const model = await loadGraphModel(customLoader);
    spyOn(console, 'warn');
    const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
    await model.executeAsync(input);
    expect(console.warn).toHaveBeenCalledTimes(1);
    expect(console.warn)
        .toHaveBeenCalledWith(
            'This model execution did not contain any nodes with control ' +
            'flow or dynamic output shapes. You can use model.execute() ' +
            'instead.');
  });

  it('executeAsync does not warn when there are dynamic ops', async () => {
    const customLoader: tfc.io.IOHandler = {
      load: async () => ({
        modelTopology: CONTROL_FLOW_MODEL,
        weightSpecs: weightsManifest,
        weightData: new Int32Array([5]).buffer,
      })
    };
    const model = await loadGraphModel(customLoader);
    spyOn(console, 'warn');
    const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
    await model.executeAsync(input);
    expect(console.warn).toHaveBeenCalledTimes(0);
  });

  it('executeAsync warns when the subgraph has no dynamic ops', async () => {
    const graphDef: tensorflow.IGraphDef = {
      node: [
        {name: 'input', op: 'Placeholder'},
        {name: 'intermediate', op: 'Enter', input: ['input']},
        {name: 'intermediate2', op: 'Sqrt', input: ['intermediate']},
        {name: 'output', op: 'Square', input: ['intermediate2']},
      ],
      versions: {producer: 1.0, minConsumer: 3}
    };
    const customLoader: tfc.io.IOHandler = {
      load: async () => ({
        modelTopology: graphDef,
        weightSpecs: weightsManifest,
        weightData: new Int32Array([5]).buffer,
      })
    };
    const model = await loadGraphModel(customLoader);
    spyOn(console, 'warn');
    const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
    await model.executeAsync({'intermediate2': input});
    expect(console.warn).toHaveBeenCalledTimes(1);
    expect(console.warn)
        .toHaveBeenCalledWith(
            'This model execution did not contain any nodes with control ' +
            'flow or dynamic output shapes. You can use model.execute() ' +
            'instead.');
  });

  it('executeAsync works when the subgraph has no unknown ops', async () => {
    const graphDef: tensorflow.IGraphDef = {
      node: [
        {name: 'input', op: 'Placeholder'},
        {name: 'intermediate', op: 'Unknown', input: ['input']},
        {name: 'intermediate2', op: 'Sqrt', input: ['intermediate']},
        {name: 'output', op: 'Square', input: ['intermediate2']},
      ],
      versions: {producer: 1.0, minConsumer: 3}
    };
    const customLoader: tfc.io.IOHandler = {
      load: async () => ({
        modelTopology: graphDef,
        weightSpecs: weightsManifest,
        weightData: new Int32Array([5]).buffer,
      })
    };
    const model = await loadGraphModel(customLoader);
    spyOn(console, 'warn');
    const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
    await model.executeAsync({'intermediate2': input});
  });

  it('executeAsync throws when the subgraph has unknown ops', async () => {
    const graphDef: tensorflow.IGraphDef = {
      node: [
        {name: 'input', op: 'Placeholder'},
        {name: 'intermediate', op: 'Unknown', input: ['input']},
        {name: 'intermediate2', op: 'Sqrt', input: ['intermediate']},
        {name: 'output', op: 'Square', input: ['intermediate2']},
      ],
      versions: {producer: 1.0, minConsumer: 3}
    };
    const customLoader: tfc.io.IOHandler = {
      load: async () => ({
        modelTopology: graphDef,
        weightSpecs: weightsManifest,
        weightData: new Int32Array([5]).buffer,
      })
    };
    const model = await loadGraphModel(customLoader);
    const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
    try {
      await model.executeAsync({'input': input});
      throw new Error('Previous line should throw');
    } catch (ex) {
      expect((ex as Error).message)
          .toBe(
              'Unknown op \'Unknown\'. File an issue at ' +
              'https://github.com/tensorflow/tfjs/issues so we can add it, ' +
              'or register a custom execution with tf.registerOp()');
    }
  });

  it('execute fails when there are dynamic ops', async () => {
    const customLoader: tfc.io.IOHandler = {
      load: async () => ({
        modelTopology: CONTROL_FLOW_MODEL,
        weightSpecs: weightsManifest,
        weightData: new Int32Array([5]).buffer,
      })
    };
    const model = await loadGraphModel(customLoader);
    const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
    expect(() => model.execute(input))
        .toThrowError(/This execution contains the node 'Enter'/);
  });

  it('execute does not warn when there are no dynamic ops', async () => {
    const customLoader: tfc.io.IOHandler = {
      load: async () => ({
        modelTopology: SIMPLE_MODEL,
        weightSpecs: weightsManifest,
        weightData: new Int32Array([5]).buffer,
      })
    };
    const model = await loadGraphModel(customLoader);
    spyOn(console, 'warn');
    const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
    model.execute(input);
    expect(console.warn).toHaveBeenCalledTimes(0);
  });

  it('execute err when there are no dynamic ops', async () => {
    const customLoader: tfc.io.IOHandler = {
      load: async () => ({
        modelTopology: SIMPLE_MODEL,
        weightSpecs: weightsManifest,
        weightData: new Int32Array([5]).buffer,
      })
    };
    const model = await loadGraphModel(customLoader);
    spyOn(console, 'warn');
    const input = tfc.tensor2d([1, 1], [2, 1], 'int32');
    model.execute(input);
    expect(console.warn).toHaveBeenCalledTimes(0);
  });
});
