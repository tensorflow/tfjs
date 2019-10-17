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
// information. The map key is TFSavedModelSignature id in JavaScript, value is
// a turple of path to the SavedModel, metagraph tags, and loaded Session ID in
// the c++ bindings. When user loads a SavedModel signature, it will go through
// entries in this map to find if the corresponding SavedModel session has
// already been loaded in C++ addon and will reuse it if existing.
const loadedSavedModelPathMap = new Map<number, [string, string, number]>();

let tfSavedModelSignatureId = 0;

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
 */
export function getInputAndOutputNodeNameFromMetaGraphInfo(
    savedModelInfo: MetaGraphInfo[], tags: string[], signature: string) {
  for (let i = 0; i < savedModelInfo.length; i++) {
    const metaGraphInfo = savedModelInfo[i];
    if (tags.length === metaGraphInfo.tags.length &&
        JSON.stringify(tags) === JSON.stringify(metaGraphInfo.tags)) {
      if (metaGraphInfo.signatureDefs[signature] === undefined) {
        throw new Error('The SavedModel does not have signature: ' + signature);
      }
      const inputNodeNames: string[] = [];
      const outputNodeNames: string[] = [];
      for (const signature of Object.keys(metaGraphInfo.signatureDefs)) {
        for (const tensorName of Object.keys(
                 metaGraphInfo.signatureDefs[signature].inputs)) {
          inputNodeNames.push(
              metaGraphInfo.signatureDefs[signature].inputs[tensorName].name);
        }
        for (const tensorName of Object.keys(
                 metaGraphInfo.signatureDefs[signature].outputs)) {
          inputNodeNames.push(
              metaGraphInfo.signatureDefs[signature].outputs[tensorName].name);
        }
      }
      return [inputNodeNames, outputNodeNames];
    }
  }
  throw new Error(`The SavedModel does not have tags: ${tags}`);
}

/**
 * A `tf.TFSavedModelSignature` is a signature loaded from a SavedModel
 * metagraph, and allows inference exeuction.
 */
export class TFSavedModelSignature implements InferenceModel {
  // ID of the loaded session in C++ bindings
  private readonly sessionId: number;
  // ID of the loaded signature in javascript
  private readonly jsid: number;
  private readonly backend: NodeJSKernelBackend;
  private deleted: boolean;

  constructor(
      sessionId: number, jsid: number, inputNodeNames: string[],
      outputNodeNames: string[], backend: NodeJSKernelBackend) {
    this.sessionId = sessionId;
    this.backend = backend;
    this.deleted = false;
    this.jsid = jsid;
  }

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
   * the C++ backend if the session is only used by this TFSavedModelSignature.
   */
  delete() {
    if (!this.deleted) {
      this.deleted = true;

      loadedSavedModelPathMap.delete(this.jsid);
      for (const id of Array.from(loadedSavedModelPathMap.keys())) {
        const value = loadedSavedModelPathMap.get(id);
        if (value[2] === this.sessionId) {
          return;
        }
      }
      this.backend.deleteSavedModel(this.sessionId);
    } else {
      throw new Error('This SavedModel has been deleted.');
    }
  }

  /**
   * Placeholder function.
   * @param inputs
   * @param config
   */
  predict(inputs: Tensor|Tensor[]|NamedTensorMap, config?: ModelPredictConfig):
      Tensor|Tensor[]|NamedTensorMap {
    if (this.deleted) {
      throw new Error('The TFSavedModelSignature has been deleted!');
    } else {
      throw new Error(
          'predict() of TFSavedModelSignature is not supported yet.');
    }
  }

  /**
   * Placeholder function.
   * @param inputs
   * @param outputs
   */
  execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs: string|string[]):
      Tensor|Tensor[] {
    throw new Error('Execute() of TFSavedModelSignature is not supported yet.');
  }
}

/**
 * Load a signature of a MetaGraph from a SavedModel as `TFSavedModelSignature`.
 * The loaded `TFSavedModelSignature` can be used to do inference execution.
 *
 * @param path The path to the SavedModel.
 * @param tags The tags of the MetaGraph to load.
 * @param signature The SignatureDef to load.
 */
export async function loadSavedModel(
    path: string, tags: string[],
    signature: string): Promise<TFSavedModelSignature> {
  ensureTensorflowBackend();
  // Convert metagraph tags string array to a string.
  const tagsString = tags.join();

  const backend = nodeBackend();

  const savedModelInfo = await getMetaGraphsFromSavedModel(path);
  const [inputNodeNames, outputNodeNames] =
      getInputAndOutputNodeNameFromMetaGraphInfo(
          savedModelInfo, tags, signature);

  let sessionId;

  for (const id of Array.from(loadedSavedModelPathMap.keys())) {
    const value = loadedSavedModelPathMap.get(id);
    if (value[0] === path && value[1] === tagsString) {
      sessionId = value[2];
    }
  }
  if (typeof sessionId === 'undefined') {
    sessionId = backend.loadSavedModelMetaGraph(path, tagsString);
  }
  const id = tfSavedModelSignatureId++;
  const modelSignature = new TFSavedModelSignature(
      sessionId, id, inputNodeNames, outputNodeNames, backend);
  loadedSavedModelPathMap.set(id, [path, tagsString, sessionId]);
  return modelSignature;
}
