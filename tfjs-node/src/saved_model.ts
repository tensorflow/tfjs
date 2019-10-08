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

import * as fs from 'fs';
import {promisify} from 'util';

const readFile = promisify(fs.readFile);

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

// tslint:disable-next-line:no-any
export async function readSavedModelProto(path: string) {
  // Load the SavedModel pb file and deserialize it into message.
  const modelFile = await readFile(path);
  const array = new Uint8Array(modelFile);
  return messages.SavedModel.deserializeBinary(array);
}

/**
 * Inspect the contents of the SavedModel from the provided path.
 *
 * @param path Path to SavedModel folder.
 */
export async function inspectSavedModel(path: string) {
  const result = [];
  const modelMessage = await readSavedModelProto(path + SAVED_MODEL_FILE_NAME);
  const metaGraphList = modelMessage.getMetaGraphsList();
  for (let i = 0; i < metaGraphList.length; i++) {
    const metaGraph = {} as MetaGraphInfo;
    const tags = metaGraphList[i].getMetaInfoDef().getTagsList();
    metaGraph.tags = tags;

    const signatureDefs = [];
    const signatureDefMap = metaGraphList[i].getSignatureDefMap();
    const signatureDefKeys = signatureDefMap.keys();
    while (true) {
      const key = signatureDefKeys.next();
      if (key.done) {
        break;
      }
      const signatureDef: SignatureDefInfo = {};
      const signatureDefEntry = signatureDefMap.get(key.value);
      // inputs
      const inputsMapMessage = signatureDefEntry.getInputsMap();
      const inputsMapKeys = inputsMapMessage.keys();
      const inputs: TensorInfo[] = [];
      while (true) {
        const inputsMapKey = inputsMapKeys.next();
        if (inputsMapKey.done) {
          break;
        }
        const inputTensor = inputsMapMessage.get(inputsMapKey.value);
        const inputTensorInfo = {} as TensorInfo;
        inputTensorInfo.dtype =
            getEnumKeyFromValue(messages.DataType, inputTensor.getDtype());
        inputTensorInfo.name = inputTensor.getName();
        inputTensorInfo.shape = inputTensor.getTensorShape().getDimList();
        inputs.push(inputTensorInfo);
      }
      // outputs
      const outputsMapMessage = signatureDefEntry.getOutputsMap();
      const outputsMapKeys = outputsMapMessage.keys();
      const outputs: TensorInfo[] = [];
      while (true) {
        const outputsMapKey = outputsMapKeys.next();
        if (outputsMapKey.done) {
          break;
        }
        const outputTensor = outputsMapMessage.get(outputsMapKey.value);
        const outputTensorInfo = {} as TensorInfo;
        outputTensorInfo.dtype =
            getEnumKeyFromValue(messages.DataType, outputTensor.getDtype());
        outputTensorInfo.name = outputTensor.getName();
        outputTensorInfo.shape = outputTensor.getTensorShape().getDimList();
        outputs.push(outputTensorInfo);
      }

      signatureDef[key.value] = {inputs, outputs};
      signatureDefs.push(signatureDef);
    }
    metaGraph.signatureDefs = signatureDefs;

    result.push(metaGraph);
  }
  return result;
}

/**
 * Interface for inspected SavedModel MetaGraph info..
 */
export interface MetaGraphInfo {
  tags: string[];
  signatureDefs: SignatureDefInfo[];
}

/**
 * Interface for inspected SavedModel SignatureDef info..
 */
export interface SignatureDefInfo {
  [key: string]: {inputs: TensorInfo[]; outputs: TensorInfo[];}
}

/**
 * Interface for inspected SavedModel signature input/output Tensor info..
 */
export interface TensorInfo {
  dtype: string;
  shape: number[];
  name: string;
}
