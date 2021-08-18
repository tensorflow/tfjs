/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {NamedTensorMap, test_util} from '@tensorflow/tfjs';
import * as tf from './index';
import {nodeBackend} from './nodejs_kernel_backend';
import {getEnumKeyFromValue, getSignatureDefEntryFromMetaGraphInfo, readSavedModelProto} from './saved_model';

// tslint:disable-next-line:no-require-imports
const messages = require('./proto/api_pb');

describe('SavedModel', () => {
  it('deserialize SavedModel pb file', async () => {
    /**
     * The SavedModel has one MetaGraph (tag is serve). Part of the MetaGraph
     * look like:
     * {
     *  MetaInfoDef: {tags: [`serve`]},
     *  signatureDef: {
     *    __saved_model_init_op: {
     *      'inputsMap': {},
     *      'outputsMap': {'__saved_model_init_op': {'name': 'NoOp', 'dtype':
     * 0}},
     *    },
     *    serving_default: {
     *      'inputs': {
     *        x: {
     *          'name': 'serving_default_x:0',
     *          'dtype': 1,
     *        }
     *      },
     *      'outputs': {
     *        output_0: {
     *          'name': 'StatefulPartitionedCall:0',
     *          'dtype': 1,
     *        }
     *      },
     *    }
     *  }
     * }
     */
    const modelMessage = await readSavedModelProto(
        './test_objects/saved_model/times_three_float');

    // This SavedModel has one MetaGraph with tag serve
    expect(modelMessage.getMetaGraphsList().length).toBe(1);
    expect(modelMessage.getMetaGraphsList()[0]
               .getMetaInfoDef()
               .getTagsList()
               .length)
        .toBe(1);
    expect(
        modelMessage.getMetaGraphsList()[0].getMetaInfoDef().getTagsList()[0])
        .toBe('serve');

    // Validate the SavedModel has signatureDef serving_default
    const signatureDefMapMessage =
        modelMessage.getMetaGraphsList()[0].getSignatureDefMap();
    expect(signatureDefMapMessage.has('serving_default'));

    // The input op of signature serving_default is serving_default_x, DataType
    // is DT_FLOAT
    const inputsMapMessage =
        signatureDefMapMessage.get('serving_default').getInputsMap();
    expect(inputsMapMessage.getLength()).toBe(1);
    const inputsMapKeys = inputsMapMessage.keys();
    const inputsMapKey1 = inputsMapKeys.next();
    expect(inputsMapKey1.done).toBe(false);
    expect(inputsMapKey1.value).toBe('x');
    const inputTensorMessage = inputsMapMessage.get(inputsMapKey1.value);
    expect(inputTensorMessage.getName()).toBe('serving_default_x:0');
    expect(
        getEnumKeyFromValue(messages.DataType, inputTensorMessage.getDtype()))
        .toBe('DT_FLOAT');

    // The output op of signature serving_default is StatefulPartitionedCall,
    // DataType is DT_FLOAT
    const outputsMapMessage =
        signatureDefMapMessage.get('serving_default').getOutputsMap();
    expect(outputsMapMessage.getLength()).toBe(1);
    const outputsMapKeys = outputsMapMessage.keys();
    const outputsMapKey1 = outputsMapKeys.next();
    expect(outputsMapKey1.done).toBe(false);
    expect(outputsMapKey1.value).toBe('output_0');
    const outputTensorMessage = outputsMapMessage.get(outputsMapKey1.value);
    expect(outputTensorMessage.getName()).toBe('StatefulPartitionedCall:0');
    expect(
        getEnumKeyFromValue(messages.DataType, outputTensorMessage.getDtype()))
        .toBe('DT_FLOAT');
  });

  it('get enum key based on value', () => {
    const DataType = messages.DataType;
    const enumKey0 = getEnumKeyFromValue(DataType, 0);
    expect(enumKey0).toBe('DT_INVALID');
    const enumKey1 = getEnumKeyFromValue(DataType, 1);
    expect(enumKey1).toBe('DT_FLOAT');
    const enumKey2 = getEnumKeyFromValue(DataType, 2);
    expect(enumKey2).toBe('DT_DOUBLE');
  });

  it('read non-exist file', async done => {
    try {
      await readSavedModelProto('/not-exist');
      done.fail();
    } catch (err) {
      expect(err.message)
          .toBe(`There is no saved_model.pb file in the directory: /not-exist`);
      done();
    }
  });

  it('inspect SavedModel metagraphs', async () => {
    const modelInfo = await tf.node.getMetaGraphsFromSavedModel(
        './test_objects/saved_model/times_three_float');
    /**
     * The inspection output should be
     * [{
     *  'tags': ['serve'],
     *  'signatureDefs': {
     *      '__saved_model_init_op': {
     *        'inputs': {},
     *        'outputs': {
     *          '__saved_model_init_op': {
     *            'dtype': 'DT_INVALID',
     *            'name': 'NoOp',
     *            'shape': []
     *          }
     *        }
     *      },
     *      'serving_default': {
     *        'inputs': {
     *          'x': {
     *            'dtype': 'DT_FLOAT',
     *            'name': 'serving_default_x:0',
     *            'shape':[]
     *          }
     *        },
     *        'outputs': {
     *          'output_0': {
     *            'dtype': 'DT_FLOAT',
     *            'name': 'StatefulPartitionedCall:0',
     *            'shape': []
     *          }
     *        }
     *      }
     *   }
     * }]
     */
    expect(modelInfo.length).toBe(1);
    expect(modelInfo[0].tags.length).toBe(1);
    expect(modelInfo[0].tags[0]).toBe('serve');
    expect(Object.keys(modelInfo[0].signatureDefs).length).toBe(1);
    expect(Object.keys(modelInfo[0].signatureDefs)[0]).toBe('serving_default');
    expect(Object.keys(modelInfo[0].signatureDefs['serving_default'].inputs)
               .length)
        .toBe(1);
    expect(modelInfo[0].signatureDefs['serving_default'].inputs['x'].name)
        .toBe('serving_default_x:0');
    expect(modelInfo[0].signatureDefs['serving_default'].inputs['x'].dtype)
        .toBe('float32');
    expect(Object.keys(modelInfo[0].signatureDefs['serving_default'].outputs)
               .length)
        .toBe(1);
    expect(
        modelInfo[0].signatureDefs['serving_default'].outputs['output_0'].name)
        .toBe('StatefulPartitionedCall:0');
    expect(
        modelInfo[0].signatureDefs['serving_default'].outputs['output_0'].dtype)
        .toBe('float32');
  });

  it('get input and output node names from SavedModel metagraphs', async () => {
    const modelInfo = await tf.node.getMetaGraphsFromSavedModel(
        './test_objects/saved_model/times_three_float');
    const signature = getSignatureDefEntryFromMetaGraphInfo(
        modelInfo, ['serve'], 'serving_default');
    expect(Object.keys(signature).length).toBe(2);
    expect(signature.inputs['x'].name).toBe('serving_default_x:0');
    expect(signature.outputs['output_0'].name)
        .toBe('StatefulPartitionedCall:0');
  });

  it('load TFSavedModel', async () => {
    const loadSavedModelMetaGraphSpy =
        spyOn(nodeBackend(), 'loadSavedModelMetaGraph').and.callThrough();
    expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(0);
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/times_three_float', ['serve'],
        'serving_default');
    expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(1);
    model.dispose();
  });

  it('load TFSavedModel with wrong tags throw exception', async done => {
    try {
      await tf.node.loadSavedModel(
          './test_objects/saved_model/times_three_float', ['serve', 'gpu'],
          'serving_default');
      done.fail();
    } catch (error) {
      expect(error.message)
          .toBe('The SavedModel does not have tags: serve,gpu');
      done();
    }
  });

  it('load TFSavedModel with wrong signature throw exception', async done => {
    try {
      await tf.node.loadSavedModel(
          './test_objects/saved_model/times_three_float', ['serve'],
          'wrong_signature');
      done.fail();
    } catch (error) {
      expect(error.message)
          .toBe('The SavedModel does not have signature: wrong_signature');
      done();
    }
  });

  it('load TFSavedModel and delete', async () => {
    expect(tf.node.getNumOfSavedModels()).toBe(0);
    const loadSavedModelMetaGraphSpy =
        spyOn(nodeBackend(), 'loadSavedModelMetaGraph').and.callThrough();
    const deleteSavedModelSpy =
        spyOn(nodeBackend(), 'deleteSavedModel').and.callThrough();
    expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(0);
    expect(deleteSavedModelSpy).toHaveBeenCalledTimes(0);
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/times_three_float', ['serve'],
        'serving_default');
    expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(1);
    expect(deleteSavedModelSpy).toHaveBeenCalledTimes(0);
    expect(tf.node.getNumOfSavedModels()).toBe(1);
    model.dispose();
    expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(1);
    expect(deleteSavedModelSpy).toHaveBeenCalledTimes(1);
    expect(tf.node.getNumOfSavedModels()).toBe(0);
  });

  it('delete TFSavedModel multiple times throw exception', async done => {
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/times_three_float', ['serve'],
        'serving_default');
    model.dispose();
    try {
      model.dispose();
      done.fail();
    } catch (error) {
      expect(error.message).toBe('This SavedModel has already been deleted.');
      done();
    }
  });

  it('load multiple signatures from the same metagraph only call binding once',
     async () => {
       expect(tf.node.getNumOfSavedModels()).toBe(0);
       const backend = nodeBackend();
       const loadSavedModelMetaGraphSpy =
           spyOn(backend, 'loadSavedModelMetaGraph').and.callThrough();
       expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(0);
       const model1 = await tf.node.loadSavedModel(
           './test_objects/saved_model/module_with_multiple_signatures',
           ['serve'], 'serving_default');
       expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(1);
       expect(tf.node.getNumOfSavedModels()).toBe(1);
       const model2 = await tf.node.loadSavedModel(
           './test_objects/saved_model/module_with_multiple_signatures',
           ['serve'], 'timestwo');
       expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(1);
       expect(tf.node.getNumOfSavedModels()).toBe(1);
       model1.dispose();
       expect(tf.node.getNumOfSavedModels()).toBe(1);
       model2.dispose();
       expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(1);
       expect(tf.node.getNumOfSavedModels()).toBe(0);
     });

  it('load signature after delete call binding', async () => {
    const backend = nodeBackend();
    const spyOnCallBindingLoad =
        spyOn(backend, 'loadSavedModelMetaGraph').and.callThrough();
    const spyOnNodeBackendDelete =
        spyOn(backend, 'deleteSavedModel').and.callThrough();
    expect(spyOnCallBindingLoad).toHaveBeenCalledTimes(0);
    expect(spyOnNodeBackendDelete).toHaveBeenCalledTimes(0);
    const model1 = await tf.node.loadSavedModel(
        './test_objects/saved_model/module_with_multiple_signatures', ['serve'],
        'serving_default');
    expect(spyOnCallBindingLoad).toHaveBeenCalledTimes(1);
    expect(spyOnNodeBackendDelete).toHaveBeenCalledTimes(0);
    model1.dispose();
    expect(spyOnNodeBackendDelete).toHaveBeenCalledTimes(1);
    expect(spyOnCallBindingLoad).toHaveBeenCalledTimes(1);
    const model2 = await tf.node.loadSavedModel(
        './test_objects/saved_model/module_with_multiple_signatures', ['serve'],
        'timestwo');
    expect(spyOnCallBindingLoad).toHaveBeenCalledTimes(2);
    expect(spyOnNodeBackendDelete).toHaveBeenCalledTimes(1);
    model2.dispose();
    expect(spyOnCallBindingLoad).toHaveBeenCalledTimes(2);
    expect(spyOnNodeBackendDelete).toHaveBeenCalledTimes(2);
  });

  it('throw error when input tensors do not match input ops', async done => {
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/times_three_float', ['serve'],
        'serving_default');
    const input1 = tf.tensor1d([1.0, 2, 3]);
    const input2 = tf.tensor1d([1.0, 2, 3]);
    try {
      model.predict([input1, input2]);
      done.fail();
    } catch (error) {
      expect(error.message)
          .toBe(
              'Length of input op names (1) does not match the ' +
              'length of input tensors (2).');
      model.dispose();
      done();
    }
  });

  it('execute model float times three', async () => {
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/times_three_float', ['serve'],
        'serving_default');
    const input = tf.tensor1d([1.0, 2, 3]);
    const output = model.predict(input) as tf.Tensor;
    expect(output.shape).toEqual(input.shape);
    expect(output.dtype).toBe(input.dtype);
    expect(output.dtype).toBe('float32');
    test_util.expectArraysClose(await output.data(), await input.mul(3).data());
    model.dispose();
  });

  it('execute model with tensor array as input', async () => {
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/times_three_float', ['serve'],
        'serving_default');
    const input = tf.tensor1d([1.0, 2, 3]);
    const outputArray = model.predict([input]) as tf.Tensor[];
    expect(outputArray.length).toBe(1);
    const output = outputArray[0];
    expect(output.shape).toEqual(input.shape);
    expect(output.dtype).toBe(input.dtype);
    expect(output.dtype).toBe('float32');
    test_util.expectArraysClose(await output.data(), [3.0, 6.0, 9.0]);
    model.dispose();
  });

  it('execute model with tensor map as input', async () => {
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/times_three_float', ['serve'],
        'serving_default');
    const input = tf.tensor1d([1.0, 2, 3]);
    const outputMap = model.predict({'x': input}) as NamedTensorMap;
    const output = outputMap['output_0'];
    expect(output.shape).toEqual(input.shape);
    expect(output.dtype).toBe(input.dtype);
    expect(output.dtype).toBe('float32');
    test_util.expectArraysClose(await output.data(), [3.0, 6.0, 9.0]);
    model.dispose();
  });

  it('execute model with wrong tensor name', async done => {
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/times_three_float', ['serve'],
        'serving_default');
    const input = tf.tensor1d([1.0, 2, 3]);
    try {
      model.predict({'xyz': input});
      done.fail();
    } catch (error) {
      expect(error.message)
          .toBe(
              'The model signatureDef input names are x, however ' +
              'the provided input names are xyz.');
      model.dispose();
      done();
    }
  });
  it('execute model with uint8 input', async () => {
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/uint8_multiply', ['serve'],
        'serving_default');
    const input = tf.scalar(3, 'int32');
    const output = model.predict(input) as tf.Tensor;
    expect(output.shape).toEqual([]);
    expect(output.dtype).toBe('int32');
    test_util.expectArraysClose(await output.data(), [18]);
    model.dispose();
  });

  it('execute model with int64 input', async () => {
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/int64_multiply', ['serve'],
        'serving_default');
    const input = tf.tensor1d([3, 4], 'int32');
    const output = model.predict(input) as tf.Tensor;
    expect(output.shape).toEqual([2]);
    expect(output.dtype).toBe('int32');
    const data = await output.data();
    expect(Number(data[0])).toEqual(18);
    expect(Number(data[1])).toEqual(24);
    model.dispose();
  });

  it('execute model int times two', async () => {
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/times_two_int', ['serve'],
        'serving_default');
    const input = tf.tensor1d([1, 2, 3], 'int32');
    const output = model.predict(input) as tf.Tensor;
    expect(output.shape).toEqual(input.shape);
    expect(output.dtype).toBe(input.dtype);
    test_util.expectArraysClose(await output.data(), [2, 4, 6]);
    model.dispose();
  });

  it('execute multiple signatures from the same model', async () => {
    const backend = nodeBackend();
    const loadSavedModelMetaGraphSpy =
        spyOn(backend, 'loadSavedModelMetaGraph').and.callThrough();
    expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(0);

    const model1 = await tf.node.loadSavedModel(
        './test_objects/saved_model/module_with_multiple_signatures', ['serve'],
        'serving_default');
    expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(1);
    const input1 = tf.tensor1d([1, 2, 3]);
    const output1 = model1.predict(input1) as tf.Tensor;
    expect(output1.shape).toEqual(input1.shape);
    expect(output1.dtype).toBe(input1.dtype);
    test_util.expectArraysClose(await output1.data(), [3.0, 6.0, 9.0]);

    expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(1);
    const model2 = await tf.node.loadSavedModel(
        './test_objects/saved_model/module_with_multiple_signatures', ['serve'],
        'timestwo');
    expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(1);
    const input2 = tf.tensor1d([1, 2, 3]);
    const output2 = model2.predict(input2) as tf.Tensor;
    expect(output2.shape).toEqual(input2.shape);
    expect(output2.dtype).toBe(input2.dtype);
    test_util.expectArraysClose(await output2.data(), [2.0, 4.0, 6.0]);

    expect(loadSavedModelMetaGraphSpy).toHaveBeenCalledTimes(1);
    model1.dispose();
    model2.dispose();
  });

  it('execute model with single input and multiple outputs', async () => {
    // This test model behaves as: f(x)=[2*x, x]
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/model_single_input_multi_output', ['serve'],
        'serving_default');
    const input = tf.tensor1d([1, 2, 3], 'int32');
    const output = model.predict(input) as tf.Tensor[];
    const output1 = output[0];
    const output2 = output[1];
    expect(output1.shape).toEqual(input.shape);
    expect(output1.dtype).toBe(input.dtype);
    expect(output2.shape).toEqual(input.shape);
    expect(output2.dtype).toBe(input.dtype);
    test_util.expectArraysClose(await output1.data(), [2, 4, 6]);
    test_util.expectArraysClose(await output2.data(), [1, 2, 3]);
    model.dispose();
  });

  it('execute model with multiple inputs and multiple outputs', async () => {
    // This test model behaves as: f(x, y)=[2*x, y]
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/model_multi_output', ['serve'],
        'serving_default');
    const input1 = tf.tensor1d([1, 2, 3], 'int32');
    const input2 = tf.tensor1d([1, 2, 3], 'int32');
    const output =
        model.predict({'x': input1, 'y': input2}) as tf.NamedTensorMap;
    const output1 = output['output_0'];
    const output2 = output['output_1'];
    expect(output1.shape).toEqual(input1.shape);
    expect(output1.dtype).toBe(input1.dtype);
    expect(output2.shape).toEqual(input2.shape);
    expect(output2.dtype).toBe(input2.dtype);
    test_util.expectArraysClose(await output1.data(), [2, 4, 6]);
    test_util.expectArraysClose(await output2.data(), [1, 2, 3]);
    model.dispose();
  });

  it('load multiple models', async () => {
    expect(tf.node.getNumOfSavedModels()).toBe(0);
    const model1 = await tf.node.loadSavedModel(
        './test_objects/saved_model/module_with_multiple_signatures', ['serve'],
        'serving_default');
    expect(tf.node.getNumOfSavedModels()).toBe(1);
    const model2 = await tf.node.loadSavedModel(
        './test_objects/saved_model/model_multi_output', ['serve'],
        'serving_default');
    expect(tf.node.getNumOfSavedModels()).toBe(2);
    model1.dispose();
    expect(tf.node.getNumOfSavedModels()).toBe(1);
    model2.dispose();
    expect(tf.node.getNumOfSavedModels()).toBe(0);
  });

  it('return inputs and outputs', async () => {
    const model = await tf.node.loadSavedModel(
        './test_objects/saved_model/model_multi_output', ['serve'],
        'serving_default');
    expect(model.inputs.length).toBe(2);
    expect(model.outputs.length).toBe(2);

    expect(model.inputs[0].name).toBe('serving_default_x');
    expect(model.inputs[0].dtype).toBe('int32');
    expect(model.inputs[0].tfDtype).toBe('DT_INT32');
    expect(model.inputs[0].shape.length).toBe(0);

    expect(model.inputs[1].name).toBe('serving_default_y');
    expect(model.inputs[1].dtype).toBe('int32');
    expect(model.inputs[1].tfDtype).toBe('DT_INT32');
    expect(model.inputs[1].shape.length).toBe(0);

    expect(model.outputs[0].name).toBe('StatefulPartitionedCall');
    expect(model.outputs[0].dtype).toBe('int32');
    expect(model.outputs[0].tfDtype).toBe('DT_INT32');
    expect(model.outputs[0].shape.length).toBe(0);

    expect(model.outputs[1].name).toBe('StatefulPartitionedCall:1');
    expect(model.outputs[1].dtype).toBe('int32');
    expect(model.outputs[1].tfDtype).toBe('DT_INT32');
    expect(model.outputs[1].shape.length).toBe(0);
    model.dispose();
  });
});
