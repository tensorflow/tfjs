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

import {InferenceModel, ModelPredictConfig, Tensor} from '@tensorflow/tfjs';
// TODO(kangyizhang): import ModelTensorInfo from '@tensorflow/tfjs-core' once
// new version is released.
// tslint:disable-next-line
import {NamedTensorMap, TensorInfo} from '@tensorflow/tfjs-core/dist/tensor_types';
import * as fs from 'fs';
import {promisify} from 'util';
import {ensureTensorflowBackend, nodeBackend, NodeJSKernelBackend} from './nodejs_kernel_backend';

const readFile = promisify(fs.readFile);

// tslint:disable-next-line:no-require-imports
const messages = require('./proto/api_pb');

const SAVED_MODEL_FILE_NAME = '/saved_model.pb';

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
// TFSavedModel can be properly reused/deleted.
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
 * Inspect the MetaGraphs of the SavedModel from the provided path.
 *
 * @param path Path to SavedModel folder.
 */
/**
 * @doc {heading: 'Models', subheading: 'SavedModel', namespace: 'node'}
 */
export async function getMetaGraphsFromSavedModel(path: string):
    Promise<MetaGraphInfo[]> {
  const result: MetaGraphInfo[] = [];

  // Get SavedModel proto message
  const modelMessage = await readSavedModelProto(path);

  // A SavedModel might have multiple MetaGraphs, identified by tags. Each
  // MetaGraph also has it's own signatureDefs.
  const metaGraphList = modelMessage.getMetaGraphsList();
  for (let i = 0; i < metaGraphList.length; i++) {
    const metaGraph = {} as MetaGraphInfo;
    const tags = metaGraphList[i].getMetaInfoDef().getTagsList();
    metaGraph.tags = tags;

    // Each MetaGraph has it's own signatureDefs map.
    const signatureDef: SignatureDefInfo = {};
    const signatureDefMap = metaGraphList[i].getSignatureDefMap();
    const signatureDefKeys = signatureDefMap.keys();

    // Go through all signatureDefs
    while (true) {
      const key = signatureDefKeys.next();
      if (key.done) {
        break;
      }
      const signatureDefEntry = signatureDefMap.get(key.value);

      // Get all input tensors information
      const inputsMapMessage = signatureDefEntry.getInputsMap();
      const inputsMapKeys = inputsMapMessage.keys();
      const inputs: {[key: string]: SavedModelTensorInfo} = {};
      while (true) {
        const inputsMapKey = inputsMapKeys.next();
        if (inputsMapKey.done) {
          break;
        }
        const inputTensor = inputsMapMessage.get(inputsMapKey.value);
        const inputTensorInfo = {} as SavedModelTensorInfo;
        inputTensorInfo.dtype =
            getEnumKeyFromValue(messages.DataType, inputTensor.getDtype());
        inputTensorInfo.name = inputTensor.getName();
        inputTensorInfo.shape = inputTensor.getTensorShape().getDimList();
        inputs[inputsMapKey.value] = inputTensorInfo;
      }

      // Get all output tensors information
      const outputsMapMessage = signatureDefEntry.getOutputsMap();
      const outputsMapKeys = outputsMapMessage.keys();
      const outputs: {[key: string]: SavedModelTensorInfo} = {};
      while (true) {
        const outputsMapKey = outputsMapKeys.next();
        if (outputsMapKey.done) {
          break;
        }
        const outputTensor = outputsMapMessage.get(outputsMapKey.value);
        const outputTensorInfo = {} as SavedModelTensorInfo;
        outputTensorInfo.dtype =
            getEnumKeyFromValue(messages.DataType, outputTensor.getDtype());
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

// TODO(kangyizhang): Remove the following interfaces and use the exported
// interfaces in tfjs-core.
/**
 * Interface for inspected SavedModel MetaGraph info.
 */
export interface MetaGraphInfo {
  tags: string[];
  signatureDefs: SignatureDefInfo;
}

/**
 * Interface for inspected SavedModel SignatureDef info.
 */
export interface SignatureDefInfo {
  [key: string]: {
    inputs: {[key: string]: SavedModelTensorInfo};
    outputs: {[key: string]: SavedModelTensorInfo};
  };
}

/**
 * Interface for inspected SavedModel signature input/output Tensor info.
 */
export interface SavedModelTensorInfo {
  dtype: string;
  shape: number[];
  name: string;
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
    savedModelInfo: MetaGraphInfo[], tags: string[], signature: string) {
  for (let i = 0; i < savedModelInfo.length; i++) {
    const metaGraphInfo = savedModelInfo[i];
    if (stringArraysHaveSameElements(tags, metaGraphInfo.tags)) {
      if (metaGraphInfo.signatureDefs[signature] == null) {
        throw new Error('The SavedModel does not have signature: ' + signature);
      }
      const inputNodeNames: string[] = [];
      const outputNodeNames: string[] = [];
      for (const signatureDef of Object.keys(metaGraphInfo.signatureDefs)) {
        if (signatureDef === signature) {
          for (const tensorName of Object.keys(
                   metaGraphInfo.signatureDefs[signature].inputs)) {
            inputNodeNames.push(
                metaGraphInfo.signatureDefs[signature].inputs[tensorName].name);
          }
          for (const tensorName of Object.keys(
                   metaGraphInfo.signatureDefs[signature].outputs)) {
            outputNodeNames.push(metaGraphInfo.signatureDefs[signature]
                                     .outputs[tensorName]
                                     .name);
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
 * metagraph, and allows inference exeuction.
 */
/**
 * @doc {heading: 'Models', subheading: 'SavedModel', namespace: 'node'}
 */
export class TFSavedModel implements InferenceModel {
  private deleted = false;

  constructor(
      private sessionId: number, private jsid: number,
      private inputNodeNames: string[], private outputNodeNames: string[],
      private backend: NodeJSKernelBackend) {}

  /** Placeholder function. */
  get inputs(): TensorInfo[] {
    throw new Error('SavedModel inputs information is not available yet.');
  }

  /** Placeholder function. */
  get outputs(): TensorInfo[] {
    throw new Error('SavedModel outputs information is not available yet.');
  }

  /**
   * Delete the SavedModel from nodeBackend and delete corresponding session in
   * the C++ backend if the session is only used by this TFSavedModel.
   */
  /** @doc {heading: 'Models', subheading: 'SavedModel'} */
  delete() {
    if (!this.deleted) {
      this.deleted = true;

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
   * Placeholder function.
   * @param inputs
   * @param config
   */
  /** @doc {heading: 'Models', subheading: 'SavedModel'} */
  predict(inputs: Tensor|Tensor[]|NamedTensorMap, config?: ModelPredictConfig):
      Tensor|Tensor[]|NamedTensorMap {
    if (this.deleted) {
      throw new Error('The TFSavedModel already has been deleted!');
    } else {
      throw new Error('predict() of TFSavedModel is not supported yet.');
    }
  }

  /**
   * Placeholder function.
   * @param inputs
   * @param outputs
   */
  /** @doc {heading: 'Models', subheading: 'SavedModel'} */
  execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs: string|string[]):
      Tensor|Tensor[] {
    throw new Error('execute() of TFSavedModel is not supported yet.');
  }
}

/**
 * Load a TensorFlow SavedModel from disk. A SavedModel is a directory
 * containing serialized signatures and the states needed to run them. For more
 * information, see this guide: https://www.tensorflow.org/guide/saved_model
 *
 * @param path The path to the SavedModel.
 * @param tags The tags of the MetaGraph to load.
 * @param signature The SignatureDef to load.
 */
/** @doc {heading: 'Models', subheading: 'SavedModel', namespace: 'node'} */
export async function loadSavedModel(
    path: string, tags: string[], signature: string): Promise<TFSavedModel> {
  ensureTensorflowBackend();

  const backend = nodeBackend();

  const savedModelInfo = await getMetaGraphsFromSavedModel(path);
  const [inputNodeNames, outputNodeNames] =
      getInputAndOutputNodeNameFromMetaGraphInfo(
          savedModelInfo, tags, signature);

  let sessionId: number;

  for (const id of Array.from(loadedSavedModelPathMap.keys())) {
    const value = loadedSavedModelPathMap.get(id);
    if (value.path === path && stringArraysHaveSameElements(value.tags, tags)) {
      sessionId = value.sessionId;
    }
  }
  if (sessionId == null) {
    // Convert metagraph tags string array to a string.
    const tagsString = tags.join();
    sessionId = backend.loadSavedModelMetaGraph(path, tagsString);
  }
  const id = nextTFSavedModelId++;
  const modelSignature =
      new TFSavedModel(sessionId, id, inputNodeNames, outputNodeNames, backend);
  loadedSavedModelPathMap.set(id, {path, tags, sessionId});
  return modelSignature;
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
