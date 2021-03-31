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

import {DataType, InferenceModel, ModelPredictConfig, ModelTensorInfo, NamedTensorMap, Rank, tensor, Tensor} from '@tensorflow/tfjs-core';

import TFWebClient, {TFWebModelRunner, TFWebModelRunnerOptions, TFWebModelRunnerTensorInfo} from './tfweb_client';

const DEFAULT_TFLITE_MODEL_RUNNER_OPTIONS: TFWebModelRunnerOptions = {
  numThreads: -1,
};

/**
 * A `tf.TFLiteModel` is built from a TFLite model flatbuffer and its
 * corresponding Interpreter.
 *
 * @doc {heading: 'Models', subheading: 'Classes'}
 */
export class TFLiteModel implements InferenceModel {
  constructor(private readonly modelRunner: TFWebModelRunner) {}

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
   * @param input The input tensors, when there is single input for the model,
   * inputs param should be a Tensor. For models with multiple inputs, inputs
   * params should be in either Tensor[] if the input order is fixed, or
   * otherwise NamedTensorMap format.
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
   * @doc {heading: 'Models', subheading: 'TFLiteModel'}
   */
  predict(
      inputs: Tensor<Rank>|Tensor<Rank>[]|NamedTensorMap,
      config: ModelPredictConfig): Tensor<Rank>|Tensor<Rank>[]|NamedTensorMap {
    const modelInputs = this.modelRunner.getInputs();
    const modelOutputs = this.modelRunner.getOutputs();
    const outputTensors: Tensor[] = [];

    // Set model inputs from the given tensors.

    // A single tensor.
    if (inputs instanceof Tensor) {
      if (modelInputs.length === 0) {
        throw new Error('The TFLite model has no inputs');
      }
      const modelInput = modelInputs[0];
      this.setModelInputFromTensor(modelInput, inputs);
    }
    // An array of tensors.
    else if (Array.isArray(inputs)) {
      if (modelInputs.length !== inputs.length) {
        throw new Error(`The size of TFLite model inputs (${
            modelInputs
                .length}) does not match the size of the input tensors (${
            inputs.length})`);
      }
      for (let i = 0; i < modelInputs.length; i++) {
        this.setModelInputFromTensor(modelInputs[i], inputs[i]);
      }
    }
    // Named tensors.
    else {
      const inputTensorNames = Object.keys(inputs);
      const modelInputMap: {[name: string]: TFWebModelRunnerTensorInfo} = {};
      modelInputs.forEach(modelInput => {
        modelInputMap[modelInput.name] = modelInput;
      });
      const modelInputNames = Object.keys(modelInputMap);
      if (!this.stringArraysHaveSameElements(
              inputTensorNames, modelInputNames)) {
        throw new Error(`The model input names are ${
            modelInputNames.join()}, however the provided input names are ${
            inputTensorNames.join()}.`);
      }
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
    for (let i = 0; i < modelOutputs.length; i++) {
      const modelOutput = modelOutputs[i];
      outputTensors.push(tensor(
          modelOutput.data(), this.getShapeFromTFLiteTensorInfo(modelOutput)));
    }
    return outputTensors;
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
   * @doc {heading: 'Models', subheading: 'TFLiteModel'}
   */
  execute(
      inputs: Tensor<Rank>|Tensor<Rank>[]|NamedTensorMap,
      outputs: string|string[]): Tensor<Rank>|Tensor<Rank>[] {
    throw new Error('execute() of TFLiteModel is not supported yet.');
  }

  private setModelInputFromTensor(
      modelInput: TFWebModelRunnerTensorInfo, tensor: Tensor) {
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

  private convertTFLiteTensorInfos(infos: TFWebModelRunnerTensorInfo[]):
      ModelTensorInfo[] {
    return infos.map(info => {
      let dtype: DataType|undefined = undefined;
      switch (info.dataType) {
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
      return {
        name: info.name,
        shape: this.getShapeFromTFLiteTensorInfo(info),
        dtype,
      };
    });
  }

  private stringArraysHaveSameElements(arrayA: string[], arrayB: string[]):
      boolean {
    if (arrayA.length === arrayB.length &&
        arrayA.sort().join() === arrayB.sort().join()) {
      return true;
    }
    return false;
  }

  private getShapeFromTFLiteTensorInfo(info: TFWebModelRunnerTensorInfo) {
    return info.shape.split(',').map(s => Number(s));
  }
}

/**
 * Load a TFLiteModel from the given model url.
 *
 * @param modelUrl The path to the model.
 * @param options Options related to model inference.
 *
 * @doc {heading: 'Models', subheading: 'TFLiteModel'}
 */
export async function loadTFLiteModel(
    modelUrl: string,
    options: TFWebModelRunnerOptions =
        DEFAULT_TFLITE_MODEL_RUNNER_OPTIONS): Promise<TFLiteModel> {
  const tfliteModelRunner =
      await TFWebClient.tfweb.TFWebModelRunner.create(modelUrl, options);
  return new TFLiteModel(tfliteModelRunner);
}
