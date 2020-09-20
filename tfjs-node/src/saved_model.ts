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

import {DataType, InferenceModel, MetaGraph, ModelPredictConfig, ModelTensorInfo, NamedTensorMap, SignatureDef, Tensor, util} from '@tensorflow/tfjs';
import * as fs from 'fs';
import {promisify} from 'util';
import {ensureTensorflowBackend, nodeBackend, NodeJSKernelBackend} from './nodejs_kernel_backend';

const readFile = promisify(fs.readFile);

// tslint:disable-next-line:no-require-imports
const messages = require('./proto/api_pb');

const SAVED_MODEL_FILE_NAME = '/saved_model.pb';

const SAVED_MODEL_INIT_OP_KEY = '__saved_model_init_op';

// This map is used to keep track of loaded SavedModel metagraph mapping
// information. The map key is TFSavedModel id in JavaScript, value is
// an object of path to the SavedModel, metagraph tags, and loaded Session ID in
// the c++ bindings. When user loads a SavedModel signature, it will go through
// entries in this map to find if the corresponding SavedModel session has
// already been loaded in C++ addon and will reuse it if existing.
const loadedSavedModelPathMap =
    new Map<number, {path: string, tags: string[], sessionId: number}>();

// The ID of loaded TFSavedModel. This ID is used to keep track of loaded
// TFSavedModel, so the loaded session in c++ bindings for the corresponding
// TFSavedModel can be properly reused/disposed.
let nextTFSavedModelId = 0;

/**
 * Get a key in an object by its value. This is used to get protobuf enum value
 * from index.
 *
 * @param object
 * @param value
 */
// tslint:disable-next-line:no-any
export function getEnumKeyFromValue(object: any, value: number): string {
  return Object.keys(object).find(key => object[key] === value);
}

/**
 * Read SavedModel proto message from path.
 *
 * @param path Path to SavedModel folder.
 */
export async function readSavedModelProto(path: string) {
  // Load the SavedModel pb file and deserialize it into message.
  try {
    fs.accessSync(path + SAVED_MODEL_FILE_NAME, fs.constants.R_OK);
  } catch (error) {
    throw new Error(
        'There is no saved_model.pb file in the directory: ' + path);
  }
  const modelFile = await readFile(path + SAVED_MODEL_FILE_NAME);
  const array = new Uint8Array(modelFile);
  return messages.SavedModel.deserializeBinary(array);
}

/**
 * Inspect the MetaGraphs of the SavedModel from the provided path. This
 * function will return an array of `MetaGraphInfo` objects.
 *
 * @param path Path to SavedModel folder.
 *
 * @doc {heading: 'Models', subheading: 'SavedModel', namespace: 'node'}
 */
export async function getMetaGraphsFromSavedModel(path: string):
    Promise<MetaGraph[]> {
  const result: MetaGraph[] = [];

  // Get SavedModel proto message
  const modelMessage = await readSavedModelProto(path);

  // A SavedModel might have multiple MetaGraphs, identified by tags. Each
  // MetaGraph also has it's own signatureDefs.
  const metaGraphList = modelMessage.getMetaGraphsList();
  for (let i = 0; i < metaGraphList.length; i++) {
    const metaGraph = {} as MetaGraph;
    const tags = metaGraphList[i].getMetaInfoDef().getTagsList();
    metaGraph.tags = tags;

    // Each MetaGraph has it's own signatureDefs map.
    const signatureDef: SignatureDef = {};
    const signatureDefMap = metaGraphList[i].getSignatureDefMap();
    const signatureDefKeys = signatureDefMap.keys();

    // Go through all signatureDefs
    while (true) {
      const key = signatureDefKeys.next();
      if (key.done) {
        break;
      }
      // Skip TensorFlow internal Signature '__saved_model_init_op'.
      if (key.value === SAVED_MODEL_INIT_OP_KEY) {
        continue;
      }
      const signatureDefEntry = signatureDefMap.get(key.value);

      // Get all input tensors information
      const inputsMapMessage = signatureDefEntry.getInputsMap();
      const inputsMapKeys = inputsMapMessage.keys();
      const inputs: {[key: string]: ModelTensorInfo} = {};
      while (true) {
        const inputsMapKey = inputsMapKeys.next();
        if (inputsMapKey.done) {
          break;
        }
        const inputTensor = inputsMapMessage.get(inputsMapKey.value);
        const inputTensorInfo = {} as ModelTensorInfo;
        inputTensorInfo.dtype = mapTFDtypeToJSDtype(
            getEnumKeyFromValue(messages.DataType, inputTensor.getDtype()));

        inputTensorInfo.name = inputTensor.getName();
        inputTensorInfo.shape = inputTensor.getTensorShape().getDimList();
        inputs[inputsMapKey.value] = inputTensorInfo;
      }

      // Get all output tensors information
      const outputsMapMessage = signatureDefEntry.getOutputsMap();
      const outputsMapKeys = outputsMapMessage.keys();
      const outputs: {[key: string]: ModelTensorInfo} = {};
      while (true) {
        const outputsMapKey = outputsMapKeys.next();
        if (outputsMapKey.done) {
          break;
        }
        const outputTensor = outputsMapMessage.get(outputsMapKey.value);
        const outputTensorInfo = {} as ModelTensorInfo;
        outputTensorInfo.dtype = mapTFDtypeToJSDtype(
            getEnumKeyFromValue(messages.DataType, outputTensor.getDtype()));
        outputTensorInfo.name = outputTensor.getName();
        outputTensorInfo.shape = outputTensor.getTensorShape().getDimList();
        outputs[outputsMapKey.value] = outputTensorInfo;
      }

      signatureDef[key.value] = {inputs, outputs};
    }
    metaGraph.signatureDefs = signatureDef;

    result.push(metaGraph);
  }
  return result;
}

/**
 * Get input and output node names from SavedModel metagraphs info. The
 * input.output node names will be used when executing a SavedModel signature.
 *
 * @param savedModelInfo The MetaGraphInfo array loaded through
 *     getMetaGraphsFromSavedModel().
 * @param tags The tags of the MetaGraph to get input/output node names from.
 * @param signature The signature to get input/output node names from.
 */
export function getInputAndOutputNodeNameFromMetaGraphInfo(
    savedModelInfo: MetaGraph[], tags: string[], signature: string) {
  for (let i = 0; i < savedModelInfo.length; i++) {
    const metaGraphInfo = savedModelInfo[i];
    if (stringArraysHaveSameElements(tags, metaGraphInfo.tags)) {
      if (metaGraphInfo.signatureDefs[signature] == null) {
        throw new Error('The SavedModel does not have signature: ' + signature);
      }
      const inputNodeNames: {[key: string]: string} = {};
      const outputNodeNames: {[key: string]: string} = {};
      for (const signatureDef of Object.keys(metaGraphInfo.signatureDefs)) {
        if (signatureDef === signature) {
          for (const tensorName of Object.keys(
                   metaGraphInfo.signatureDefs[signature].inputs)) {
            inputNodeNames[tensorName] =
                metaGraphInfo.signatureDefs[signature].inputs[tensorName].name;
          }
          for (const tensorName of Object.keys(
                   metaGraphInfo.signatureDefs[signature].outputs)) {
            outputNodeNames[tensorName] =
                metaGraphInfo.signatureDefs[signature].outputs[tensorName].name;
          }
        }
      }
      return [inputNodeNames, outputNodeNames];
    }
  }
  throw new Error(`The SavedModel does not have tags: ${tags}`);
}

/**
 * A `tf.TFSavedModel` is a signature loaded from a SavedModel
 * metagraph, and allows inference execution.
 *
 * @doc {heading: 'Models', subheading: 'SavedModel', namespace: 'node'}
 */
export class TFSavedModel implements InferenceModel {
  private disposed = false;

  constructor(
      private sessionId: number, private jsid: number,
      private inputNodeNames: {[key: string]: string},
      private outputNodeNames: {[key: string]: string},
      private backend: NodeJSKernelBackend) {}

  /**
   * Return the array of input tensor info.
   *
   * @doc {heading: 'Models', subheading: 'SavedModel'}
   */
  get inputs(): ModelTensorInfo[] {
    throw new Error('SavedModel inputs information is not available yet.');
  }

  /**
   * Return the array of output tensor info.
   *
   * @doc {heading: 'Models', subheading: 'SavedModel'}
   */
  get outputs(): ModelTensorInfo[] {
    throw new Error('SavedModel outputs information is not available yet.');
  }

  /**
   * Delete the SavedModel from nodeBackend and delete corresponding session in
   * the C++ backend if the session is only used by this TFSavedModel.
   *
   * @doc {heading: 'Models', subheading: 'SavedModel'}
   */
  dispose() {
    if (!this.disposed) {
      this.disposed = true;

      loadedSavedModelPathMap.delete(this.jsid);
      for (const id of Array.from(loadedSavedModelPathMap.keys())) {
        const value = loadedSavedModelPathMap.get(id);
        if (value.sessionId === this.sessionId) {
          return;
        }
      }
      this.backend.deleteSavedModel(this.sessionId);
    } else {
      throw new Error('This SavedModel has already been deleted.');
    }
  }

  /**
   * Execute the inference for the input tensors.
   *
   * @param input The input tensors, when there is single input for the model,
   * inputs param should be a Tensor. For models with multiple inputs, inputs
   * params should be in either Tensor[] if the input order is fixed, or
   * otherwise NamedTensorMap format. The keys in the NamedTensorMap are the
   * name of input tensors in SavedModel signatureDef. It can be found through
   * `tf.node.getMetaGraphsFromSavedModel()`.
   *
   * For batch inference execution, the tensors for each input need to be
   * concatenated together. For example with mobilenet, the required input shape
   * is [1, 244, 244, 3], which represents the [batch, height, width, channel].
   * If we are provide a batched data of 100 images, the input tensor should be
   * in the shape of [100, 244, 244, 3].
   *
   * @param config Prediction configuration for specifying the batch size.
   *
   * @returns Inference result tensors. The output would be single Tensor if
   * model has single output node, otherwise Tensor[] or NamedTensorMap[] will
   * be returned for model with multiple outputs.
   *
   * @doc {heading: 'Models', subheading: 'SavedModel'}
   */
  predict(inputs: Tensor|Tensor[]|NamedTensorMap, config?: ModelPredictConfig):
      Tensor|Tensor[]|NamedTensorMap {
    if (this.disposed) {
      throw new Error('The TFSavedModel has already been deleted!');
    } else {
      let inputTensors: Tensor[] = [];
      if (inputs instanceof Tensor) {
        inputTensors.push(inputs);
        const result = this.backend.runSavedModel(
            this.sessionId, inputTensors, Object.values(this.inputNodeNames),
            Object.values(this.outputNodeNames));
        return result.length > 1 ? result : result[0];
      } else if (Array.isArray(inputs)) {
        inputTensors = inputs;
        return this.backend.runSavedModel(
            this.sessionId, inputTensors, Object.values(this.inputNodeNames),
            Object.values(this.outputNodeNames));
      } else {
        const inputTensorNames = Object.keys(this.inputNodeNames);
        const providedInputNames = Object.keys(inputs);
        if (!stringArraysHaveSameElements(
                inputTensorNames, providedInputNames)) {
          throw new Error(`The model signatureDef input names are ${
              inputTensorNames.join()}, however the provided input names are ${
              providedInputNames.join()}.`);
        }
        const inputNodeNamesArray = [];
        for (let i = 0; i < inputTensorNames.length; i++) {
          inputTensors.push(inputs[inputTensorNames[i]]);
          inputNodeNamesArray.push(this.inputNodeNames[inputTensorNames[i]]);
        }
        const outputTensorNames = Object.keys(this.outputNodeNames);
        const outputNodeNamesArray = [];
        for (let i = 0; i < outputTensorNames.length; i++) {
          outputNodeNamesArray.push(this.outputNodeNames[outputTensorNames[i]]);
        }
        const outputTensors = this.backend.runSavedModel(
            this.sessionId, inputTensors, inputNodeNamesArray,
            outputNodeNamesArray);
        util.assert(
            outputTensors.length === outputNodeNamesArray.length,
            () => 'Output tensors do not match output node names, ' +
                `receive ${outputTensors.length}) output tensors but ` +
                `there are ${this.outputNodeNames.length} output nodes.`);
        const outputMap: NamedTensorMap = {};
        for (let i = 0; i < outputTensorNames.length; i++) {
          outputMap[outputTensorNames[i]] = outputTensors[i];
        }
        return outputMap;
      }
    }
  }

  /**
   * Execute the inference for the input tensors and return activation
   * values for specified output node names without batching.
   *
   * @param input The input tensors, when there is single input for the model,
   * inputs param should be a Tensor. For models with multiple inputs, inputs
   * params should be in either Tensor[] if the input order is fixed, or
   * otherwise NamedTensorMap format.
   *
   * @param outputs string|string[]. List of output node names to retrieve
   * activation from.
   *
   * @returns Activation values for the output nodes result tensors. The return
   * type matches specified parameter outputs type. The output would be single
   * Tensor if single output is specified, otherwise Tensor[] for multiple
   * outputs.
   *
   * @doc {heading: 'Models', subheading: 'SavedModel'}
   */
  execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs: string|string[]):
      Tensor|Tensor[] {
    throw new Error('execute() of TFSavedModel is not supported yet.');
  }
}

/**
 * Load a TensorFlow SavedModel from disk. TensorFlow SavedModel is different
 * from TensorFlow.js model format. A SavedModel is a directory containing
 * serialized signatures and the states needed to run them. The directory has a
 * saved_model.pb (or saved_model.pbtxt) file storing the actual TensorFlow
 * program, or model, and a set of named signatures, each identifying a
 * function. The directory also has a variables directory contains a standard
 * training checkpoint. The directory may also has a assets directory contains
 * files used by the TensorFlow graph, for example text files used to initialize
 * vocabulary tables. These are supported datatypes: float32, int32, complex64,
 * string.For more information, see this guide:
 * https://www.tensorflow.org/guide/saved_model.
 *
 * @param path The path to the SavedModel.
 * @param tags The tags of the MetaGraph to load. The available tags of a
 *     SavedModel can be retrieved through tf.node.getMetaGraphsFromSavedModel()
 *     API. Defaults to ['serve'].
 * @param signature The name of the SignatureDef to load. The available
 *     SignatureDefs of a SavedModel can be retrieved through
 *     tf.node.getMetaGraphsFromSavedModel() API. Defaults to 'serving_default'.
 *
 * @doc {heading: 'Models', subheading: 'SavedModel', namespace: 'node'}
 */
export async function loadSavedModel(
    path: string, tags = ['serve'],
    signature = 'serving_default'): Promise<TFSavedModel> {
  ensureTensorflowBackend();

  const backend = nodeBackend();

  const savedModelInfo = await getMetaGraphsFromSavedModel(path);
  const [inputNodeNames, outputNodeNames] =
      getInputAndOutputNodeNameFromMetaGraphInfo(
          savedModelInfo, tags, signature);

  let sessionId: number;

  for (const id of Array.from(loadedSavedModelPathMap.keys())) {
    const modelInfo = loadedSavedModelPathMap.get(id);
    if (modelInfo.path === path &&
        stringArraysHaveSameElements(modelInfo.tags, tags)) {
      sessionId = modelInfo.sessionId;
    }
  }
  if (sessionId == null) {
    // Convert metagraph tags string array to a string.
    const tagsString = tags.join(',');
    sessionId = backend.loadSavedModelMetaGraph(path, tagsString);
  }
  const id = nextTFSavedModelId++;
  const savedModel =
      new TFSavedModel(sessionId, id, inputNodeNames, outputNodeNames, backend);
  loadedSavedModelPathMap.set(id, {path, tags, sessionId});
  return savedModel;
}

/**
 * Compare if two unsorted arrays of string have the same elements.
 * @param arrayA
 * @param arrayB
 */
function stringArraysHaveSameElements(
    arrayA: string[], arrayB: string[]): boolean {
  if (arrayA.length === arrayB.length &&
      arrayA.sort().join() === arrayB.sort().join()) {
    return true;
  }
  return false;
}

function mapTFDtypeToJSDtype(tfDtype: string): DataType {
  switch (tfDtype) {
    case 'DT_FLOAT':
      return 'float32';
    case 'DT_INT32':
      return 'int32';
    case 'DT_BOOL':
      return 'bool';
    case 'DT_COMPLEX64':
      return 'complex64';
    case 'DT_STRING':
      return 'string';
    default:
      throw new Error(
          'Unsupported tensor DataType: ' + tfDtype +
          ', try to modify the model in python to convert the datatype');
  }
}

export function getNumOfSavedModels() {
  ensureTensorflowBackend();
  const backend = nodeBackend();
  return backend.getNumOfSavedModels();
}
