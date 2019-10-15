/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {nodeBackend} from './nodejs_kernel_backend';
import {getEnumKeyFromValue, inspectSavedModel, loadSavedModel, readSavedModelProto} from './saved_model';

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
    const modelMessage =
        await readSavedModelProto('./test_objects/times_three_float');

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

  it('inspect SavedModel', async () => {
    const modelInfo =
        await inspectSavedModel('./test_objects/times_three_float');
    /**
     * The inspection output should be
     * [{
     *  'tags': ['serve'],
     *  'signatureDefs': {
     *      '__saved_model_init_op': {
     *        'inputs': [],
     *        'outputs': [{'dtype': 'DT_INVALID', 'name': 'NoOp', 'shape': []}]
     *      },
     *      'serving_default': {
     *        'inputs': [
     *          {'dtype': 'DT_FLOAT', 'name': 'serving_default_x:0', 'shape':
     * []}
     *        ],
     *        'outputs': [{
     *          'dtype': 'DT_FLOAT',
     *          'name': 'StatefulPartitionedCall:0',
     *          'shape': []
     *        }]
     *      }
     *   }
     * }]
     */
    expect(modelInfo.length).toBe(1);
    expect(modelInfo[0].tags.length).toBe(1);
    expect(modelInfo[0].tags[0]).toBe('serve');
    expect(Object.keys(modelInfo[0].signatureDefs).length).toBe(2);
    expect(Object.keys(modelInfo[0].signatureDefs)[0])
        .toBe('__saved_model_init_op');
    expect(Object.keys(modelInfo[0].signatureDefs)[1]).toBe('serving_default');
    expect(modelInfo[0].signatureDefs['serving_default'].inputs.length).toBe(1);
    expect(modelInfo[0].signatureDefs['serving_default'].inputs[0].name)
        .toBe('serving_default_x:0');
    expect(modelInfo[0].signatureDefs['serving_default'].inputs[0].dtype)
        .toBe('DT_FLOAT');
    expect(modelInfo[0].signatureDefs['serving_default'].outputs.length)
        .toBe(1);
    expect(modelInfo[0].signatureDefs['serving_default'].outputs[0].name)
        .toBe('StatefulPartitionedCall:0');
    expect(modelInfo[0].signatureDefs['serving_default'].outputs[0].dtype)
        .toBe('DT_FLOAT');
  });

  it('load TFSavedModelSignature', async () => {
    const spy = spyOn(nodeBackend(), 'loadSavedModel').and.callThrough();
    const model = await loadSavedModel(
        './test_objects/times_three_float', ['serve'], 'serving_default');
    expect(spy).toHaveBeenCalledTimes(1);
    model.delete();
  });

  it('load TFSavedModelSignature with wrong tags throw exception',
     async done => {
       try {
         await loadSavedModel(
             './test_objects/times_three_float', ['serve', 'gpu'],
             'serving_default');
         done.fail();
       } catch (error) {
         expect(error.message)
             .toBe('The SavedModel does not have tags: serve,gpu');
         done();
       }
     });

  it('load TFSavedModelSignature with wrong signature throw exception',
     async done => {
       try {
         await loadSavedModel(
             './test_objects/times_three_float', ['serve'], 'wrong_signature');
         done.fail();
       } catch (error) {
         expect(error.message)
             .toBe('The SavedModel does not have signature: wrong_signature');
         done();
       }
     });

  it('load TFSavedModelSignature and delete', async () => {
    const spy = spyOn(nodeBackend(), 'loadSavedModel').and.callThrough();
    const spy1 = spyOn(nodeBackend(), 'deleteSavedModel').and.callThrough();
    const model = await loadSavedModel(
        './test_objects/times_three_float', ['serve'], 'serving_default');
    expect(spy).toHaveBeenCalledTimes(1);
    model.delete();
    expect(spy1).toHaveBeenCalledTimes(1);
  });

  it('delete TFSavedModelSignature multiple times throw exception',
     async done => {
       try {
         const model = await loadSavedModel(
             './test_objects/times_three_float', ['serve'], 'serving_default');
         model.delete();
         model.delete();
         done.fail();
       } catch (error) {
         expect(error.message).toBe('This SavedModel has been deleted.');
         done();
       }
     });

  it('load multiple signatures from the same metagraph only call binding once',
     async () => {
       const backend = nodeBackend();
       const spy = spyOn(backend, 'loadSavedModel').and.callThrough();
       const spy1 = spyOn(backend, 'loadMetaGraph').and.callThrough();
       const signature1 = await loadSavedModel(
           './test_objects/module_with_multiple_signatures', ['serve'],
           'serving_default');
       expect(spy).toHaveBeenCalledTimes(1);
       expect(spy1).toHaveBeenCalledTimes(1);
       const signature2 = await loadSavedModel(
           './test_objects/module_with_multiple_signatures', ['serve'],
           'timestwo');
       expect(spy).toHaveBeenCalledTimes(2);
       expect(spy1).toHaveBeenCalledTimes(1);
       signature1.delete();
       signature2.delete();
     });

  it('load signature after delete call binding', async () => {
    const backend = nodeBackend();
    const spyOnNodeBackendLoad =
        spyOn(backend, 'loadSavedModel').and.callThrough();
    const spyOnCallBindingLoad =
        spyOn(backend, 'loadMetaGraph').and.callThrough();
    const spyOnNodeBackendDelete =
        spyOn(backend, 'deleteSavedModel').and.callThrough();
    const signature1 = await loadSavedModel(
        './test_objects/module_with_multiple_signatures', ['serve'],
        'serving_default');
    expect(spyOnNodeBackendLoad).toHaveBeenCalledTimes(1);
    expect(spyOnCallBindingLoad).toHaveBeenCalledTimes(1);
    signature1.delete();
    expect(spyOnNodeBackendDelete).toHaveBeenCalledTimes(1);
    const signature2 = await loadSavedModel(
        './test_objects/module_with_multiple_signatures', ['serve'],
        'timestwo');
    expect(spyOnNodeBackendLoad).toHaveBeenCalledTimes(2);
    expect(spyOnCallBindingLoad).toHaveBeenCalledTimes(2);
    signature2.delete();
  });
});
