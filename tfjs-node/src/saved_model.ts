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
import {NamedTensorMap} from '@tensorflow/tfjs-converter/dist/src/data/types';
import {TensorInfo} from '@tensorflow/tfjs-core/dist/tensor_types';
import * as fs from 'fs';
import {promisify} from 'util';
import {ensureTensorflowBackend, nodeBackend, NodeJSKernelBackend} from './nodejs_kernel_backend';

const readFile = promisify(fs.readFile);

let jsid = 0;

// tslint:disable-next-line:no-require-imports
const messages = require('./proto/api_pb');

const SAVED_MODEL_FILE_NAME = '/saved_model.pb';

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
 * Inspect the contents of the SavedModel from the provided path.
 *
 * @param path Path to SavedModel folder.
 */
export async function inspectSavedModel(path: string):
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
      const inputs: SavedModelTensorInfo[] = [];
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
        inputs.push(inputTensorInfo);
      }

      // Get all output tensors information
      const outputsMapMessage = signatureDefEntry.getOutputsMap();
      const outputsMapKeys = outputsMapMessage.keys();
      const outputs: SavedModelTensorInfo[] = [];
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
        outputs.push(outputTensorInfo);
      }

      signatureDef[key.value] = {inputs, outputs};
    }
    metaGraph.signatureDefs = signatureDef;

    result.push(metaGraph);
  }
  return result;
}

export function getInputAndOutputNodeNameFromSavedModelInfo(
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
      for (const key of Object.keys(metaGraphInfo.signatureDefs)) {
        metaGraphInfo.signatureDefs[key].inputs.map(tensorInfo => {
          inputNodeNames.push(tensorInfo.name);
        });
        metaGraphInfo.signatureDefs[key].outputs.map(tensorInfo => {
          outputNodeNames.push(tensorInfo.name);
        });
      }
      return [inputNodeNames, outputNodeNames];
    }
  }
  throw new Error('The SavedModel does not have tags: ' + tags);
}

/**
 * Interface for inspected SavedModel MetaGraph info..
 */
export interface MetaGraphInfo {
  tags: string[];
  signatureDefs: SignatureDefInfo;
}

/**
 * Interface for inspected SavedModel SignatureDef info..
 */
export interface SignatureDefInfo {
  [key: string]:
      {inputs: SavedModelTensorInfo[]; outputs: SavedModelTensorInfo[];};
}

/**
 * Interface for inspected SavedModel signature input/output Tensor info..
 */
export interface SavedModelTensorInfo {
  dtype: string;
  shape: number[];
  name: string;
}


export class TFSavedModelSignature implements InferenceModel {
  // ID of the loaded session in bindings
  private readonly cid: number;
  // ID of the object in javascript
  private readonly jsid: number;
  private readonly backend: NodeJSKernelBackend;
  private readonly path: string;
  private readonly inputNodeNames: string[];
  private readonly outputNodeNames: string[];
  private deleted: boolean;

  constructor(
      cid: number, path: string, inputNodeNames: string[],
      outputNodeNames: string[], backend: NodeJSKernelBackend) {
    this.cid = cid;
    this.path = path;
    this.inputNodeNames = inputNodeNames;
    this.outputNodeNames = outputNodeNames;
    this.backend = backend;
    this.deleted = false;
    this.jsid = jsid++;
  }

  getJsid() {
    return this.jsid;
  }


  getPath() {
    return this.path;
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
   * Delete the SavedModel from nodeBackend and delete corresponding object in
   * the C++ backend.
   */
  delete() {
    if (!this.deleted) {
      this.deleted = true;
      this.backend.deleteSavedModel(this.jsid, this.cid, this.path);
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
    throw new Error(
        'predict() function of TFSavedModel is not implemented yet.');
  }

  /**
   * Placeholder function.
   * @param inputs
   * @param outputs
   */
  execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs: string|string[]):
      Tensor|Tensor[] {
    throw new Error(
        'Execute() function of TFSavedModel is not implemented yet.');
  }
}

/**
 * Decode a JPEG-encoded image to a 3D Tensor of dtype `int32`.
 *
 * @param path The path of the exported SavedModel
 */
/**
 * @doc {heading: 'SavedModel', namespace: 'node'}
 */
export async function loadSavedModel(
    path: string, tags: string[],
    signature: string): Promise<TFSavedModelSignature> {
  ensureTensorflowBackend();
  return nodeBackend().loadSavedModel(path, tags, signature);
}
