/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {DataType, InferenceModel, ModelPredictConfig, ModelTensorInfo, NamedTensorMap, tensor, Tensor} from '@tensorflow/tfjs-core';

import * as tfliteWebAPIClient from './tflite_web_api_client';
import {ProfileItem, TFLiteDataType, TFLiteWebModelRunner, TFLiteWebModelRunnerOptions, TFLiteWebModelRunnerTensorInfo} from './types/tflite_web_model_runner';

const TFHUB_SEARCH_PARAM = '?lite-format=tflite';

/**
 * A `tflite.TFLiteModel` is built from a TFLite model flatbuffer and executable
 * on TFLite interpreter. To load it, use the `loadTFLiteModel` function below.
 *
 * Sample usage:
 *
 * ```js
 * // Load the MobilenetV2 tflite model from tfhub.
 * const tfliteModel = await tflite.loadTFLiteModel(
 *     'https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/metadata/1');
 *
 * const outputTensor = tf.tidy(() => {
 *    // Get pixels data from an image.
 *    let img = tf.browser.fromPixels(document.querySelector('img'));
 *    // Resize and normalize:
 *    img = tf.image.resizeBilinear(img, [224, 224]);
 *    img = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);
 *    // Run the inference.
 *    let outputTensor = tfliteModel.predict(img);
 *    // De-normalize the result.
 *    return tf.mul(tf.add(outputTensor, 1), 127.5)
 *  });
 * console.log(outputTensor);
 *
 * ```
 *
 * @doc {heading: 'Models', subheading: 'Classes'}
 */
export class TFLiteModel implements InferenceModel {
  constructor(private readonly modelRunner: TFLiteWebModelRunner) {}

  get inputs(): ModelTensorInfo[] {
    const modelInputs = this.modelRunner.getInputs();
    return this.convertTFLiteTensorInfos(modelInputs);
  }

  get outputs(): ModelTensorInfo[] {
    const modelOutputs = this.modelRunner.getOutputs();
    return this.convertTFLiteTensorInfos(modelOutputs);
  }

  /**
   * Execute the inference for the input tensors.
   *
   * @param inputs The input tensors, when there is single input for the model,
   *     inputs param should be a Tensor. For models with multiple inputs,
   *     inputs params should be in either Tensor[] if the input order is fixed,
   *     or otherwise NamedTensorMap format.
   *
   * @param config Prediction configuration for specifying the batch size.
   *     Currently this field is not used, and batch inference is not supported.
   *
   * @returns Inference result tensors. The output would be single Tensor if
   *     model has single output node, otherwise NamedTensorMap will be returned
   *     for model with multiple outputs. Tensor[] is not used.
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  predict(inputs: Tensor|Tensor[]|NamedTensorMap, config?: ModelPredictConfig):
      Tensor|Tensor[]|NamedTensorMap {
    const modelInputs = this.modelRunner.getInputs();
    const modelOutputs = this.modelRunner.getOutputs();

    // Set model inputs from the given tensors.

    // A single tensor or a tensor array.
    if (inputs instanceof Tensor || Array.isArray(inputs)) {
      let inputTensors: Tensor[];
      if (inputs instanceof Tensor) {
        inputTensors = [inputs];
      } else {
        inputTensors = inputs;
      }
      if (modelInputs.length !== inputTensors.length) {
        throw new Error(`The size of TFLite model inputs (${
            modelInputs
                .length}) does not match the size of the input tensors (${
            inputTensors.length})`);
      }
      for (let i = 0; i < modelInputs.length; i++) {
        this.setModelInputFromTensor(modelInputs[i], inputTensors[i]);
      }
    }
    // Named tensors.
    else {
      const inputTensorNames = Object.keys(inputs);
      const modelInputMap:
          {[name: string]: TFLiteWebModelRunnerTensorInfo} = {};
      modelInputs.forEach(modelInput => {
        modelInputMap[modelInput.name] = modelInput;
      });
      const modelInputNames = Object.keys(modelInputMap);
      this.checkMapInputs(inputTensorNames, modelInputNames);
      for (const name of inputTensorNames) {
        this.setModelInputFromTensor(modelInputMap[name], inputs[name]);
      }
    }

    // Run inference.
    const success = this.modelRunner.infer();
    if (!success) {
      throw new Error('Failed running inference');
    }

    // Convert model outputs to tensors.
    const outputTensors: NamedTensorMap = {};
    for (let i = 0; i < modelOutputs.length; i++) {
      const modelOutput = modelOutputs[i];
      let data = modelOutput.data();

      // Convert TFLite tensor types that are not supported by TFJS to
      // compatible types.
      switch (modelOutput.dataType) {
        case 'int8':
        case 'int16':
        case 'uint32':
          data = Int32Array.from(data);
          break;
        case 'float64':
          console.warn(
              `WARNING: converting output tensor from 'float64' to 'float32'`);
          data = Float32Array.from(data);
          break;
        default:
          break;
      }
      const outputTensor =
          tensor(data, this.getShapeFromTFLiteTensorInfo(modelOutput));
      outputTensors[modelOutput.name] = outputTensor;
    }
    const names = Object.keys(outputTensors);
    return names.length === 1 ? outputTensors[names[0]] : outputTensors;
  }

  /**
   * Execute the inference for the input tensors and return activation
   * values for specified output node names without batching.
   *
   * @param inputs The input tensors, when there is single input for the model,
   *     inputs param should be a Tensor. For models with multiple inputs,
   *     inputs params should be in either Tensor[] if the input order is fixed,
   *     or otherwise NamedTensorMap format.
   *
   * @param outputs string|string[]. List of output node names to retrieve
   *     activation from.
   *
   * @returns Activation values for the output nodes result tensors. The return
   *     type matches specified parameter outputs type. The output would be
   *     single Tensor if single output is specified, otherwise Tensor[] for
   *     multiple outputs.
   */
  execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs: string|string[]):
      Tensor|Tensor[] {
    throw new Error('execute() of TFLiteModel is not supported yet.');
  }

  getProfilingResults(): ProfileItem[] {
    return this.modelRunner.getProfilingResults();
  }

  getProfilingSummary(): string {
    return this.modelRunner.getProfilingSummary();
  }

  private setModelInputFromTensor(
      modelInput: TFLiteWebModelRunnerTensorInfo, tensor: Tensor) {
    // String and complex tensors are not supported.
    if (tensor.dtype === 'string' || tensor.dtype === 'complex64') {
      throw new Error(`Data type '${tensor.dtype}' not supported.`);
    }

    // Check shape.
    //
    // At this point, we've already checked that input tensors and model inputs
    // have the same size.
    const modelInputShape = modelInput.shape.split(',').map(dim => Number(dim));
    if (!tensor.shape.every(
            (dim, index) => modelInputShape[index] === -1 ||
                modelInputShape[index] === dim)) {
      throw new Error(`Input tensor shape mismatch: expect '${
          modelInput.shape}', got '${tensor.shape.join(',')}'.`);
    }

    // Check types.
    switch (modelInput.dataType) {
      // All 'bool' and 'int' tflite types accpet 'bool' or 'int32' tfjs types.
      // Will throw error for 'float32' tfjs type.
      case 'bool':
      case 'int8':
      case 'uint8':
      case 'int16':
      case 'uint32':
      case 'int32':
        if (tensor.dtype === 'float32') {
          throw this.getDataTypeMismatchError(
              modelInput.dataType, tensor.dtype);
        } else if (modelInput.dataType !== tensor.dtype) {
          console.warn(`WARNING: converting '${tensor.dtype}' to '${
              modelInput.dataType}'`);
        }
        break;
      // All 'float' tflite types accept all tfjs types.
      case 'float32':
      case 'float64':
        if (modelInput.dataType !== tensor.dtype) {
          console.warn(`WARNING: converting '${tensor.dtype}' to '${
              modelInput.dataType}'`);
        }
        break;
      default:
        break;
    }

    const modelInputBuffer = modelInput.data();
    switch (modelInput.dataType) {
      case 'int8':
        modelInputBuffer.set(Int8Array.from(tensor.dataSync()));
        break;
      case 'uint8':
      case 'bool':
        modelInputBuffer.set(Uint8Array.from(tensor.dataSync()));
        break;
      case 'int16':
        modelInputBuffer.set(Int16Array.from(tensor.dataSync()));
        break;
      case 'int32':
        modelInputBuffer.set(Int32Array.from(tensor.dataSync()));
        break;
      case 'uint32':
        modelInputBuffer.set(Uint32Array.from(tensor.dataSync()));
        break;
      case 'float32':
        modelInputBuffer.set(Float32Array.from(tensor.dataSync()));
        break;
      case 'float64':
        modelInputBuffer.set(Float64Array.from(tensor.dataSync()));
        break;
      default:
        break;
    }
  }

  private convertTFLiteTensorInfos(infos: TFLiteWebModelRunnerTensorInfo[]):
      ModelTensorInfo[] {
    return infos.map(info => {
      const dtype = getDTypeFromTFLiteType(info.dataType);
      return {
        name: info.name,
        shape: this.getShapeFromTFLiteTensorInfo(info),
        dtype,
      };
    });
  }

  private checkMapInputs(
      inputTensorNames: string[], modelInputNames: string[]) {
    const notInModel =
        inputTensorNames.filter(name => !modelInputNames.includes(name));
    const notInInput =
        modelInputNames.filter(name => !inputTensorNames.includes(name));
    if (notInModel.length === 0 && notInInput.length === 0) {
      return;
    }

    const msgParts =
        ['The model input names don\'t match the model input names.'];
    if (notInModel.length > 0) {
      msgParts.push(`Names in input but missing in model: [${notInModel}].`);
    }
    if (notInInput.length > 0) {
      msgParts.push(`Names in model but missing in inputs: [${notInInput}].`);
    }
    throw new Error(msgParts.join(' '));
  }

  private getShapeFromTFLiteTensorInfo(info: TFLiteWebModelRunnerTensorInfo) {
    return info.shape.split(',').map(s => Number(s));
  }

  private getDataTypeMismatchError(expected: string, got: string) {
    return new Error(
        `Data type mismatch: input tensor expects '${expected}', got '${got}'`);
  }
}

/**
 * Loads a TFLiteModel from the given model url.
 *
 * @param model The path to the model (string), or the model content in memory
 *     (ArrayBuffer).
 * @param options Options related to model inference.
 *
 * @doc {heading: 'Models', subheading: 'Loading'}
 */
export async function loadTFLiteModel(
    model: string|ArrayBuffer,
    options?: TFLiteWebModelRunnerOptions): Promise<TFLiteModel> {
  // Handle tfhub links.
  if (typeof model === 'string' && model.includes('tfhub.dev') &&
      model.includes('lite-model') && !model.endsWith(TFHUB_SEARCH_PARAM)) {
    model = `${model}${TFHUB_SEARCH_PARAM}`;
  }

  const tfliteModelRunner =
      await tfliteWebAPIClient.tfweb.TFLiteWebModelRunner.create(
          model, options);
  return new TFLiteModel(tfliteModelRunner);
}

/**
 * Returns the compatible tfjs DataType from the given TFLite data type.
 *
 * @param tfliteType The type in TFLite.
 *
 * @doc {heading: 'Models', subheading: 'Utilities'}
 */
export function getDTypeFromTFLiteType(tfliteType: TFLiteDataType): DataType {
  let dtype: DataType;
  switch (tfliteType) {
    case 'float32':
    case 'float64':
      dtype = 'float32';
      break;
    case 'int8':
    case 'uint8':
    case 'int16':
    case 'int32':
    case 'uint32':
      dtype = 'int32';
      break;
    case 'bool':
      dtype = 'bool';
      break;
    default:
      break;
  }
  return dtype;
}
