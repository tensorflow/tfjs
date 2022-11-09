/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {GraphModel, loadGraphModel, loadGraphModelSync} from '@tensorflow/tfjs-converter';
import {InferenceModel, io, ModelPredictConfig, NamedTensorMap, Tensor} from '@tensorflow/tfjs-core';
import {setAssets} from './register_tfdf_ops';

import {TFDFLoadHandler, TFDFLoadHandlerSync} from './types/tfdf_io';

/**
 * To load a `tfdf.TFDFModel`, use the `loadTFDFModel` function below.
 *
 * Sample usage:
 *
 * ```js
 * // Load the test TFDF model.
 * const tfdfModel = await tfdf.loadTFDFModel(
 *     'https://storage.googleapis.com/tfjs-testing/adult_gbt_no_regex/model.json');
 * const inputs = {
 *   'age': tf.tensor1d([39, 40, 40, 35], 'int32'),
 *   'workclass': tf.tensor1d(
 *       ['State-gov', 'Private', 'Private', 'Federal-gov'], 'string'),
 *   'fnlwgt': tf.tensor1d([77516, 121772, 193524, 76845], 'int32'),
 *   'education':
 *      tf.tensor1d(['Bachelors', 'Assoc-voc', 'Doctorate', '9th'], 'string'),
 *   'education_num': tf.tensor1d([13, 11, 16, 5], 'int32'),
 *   'marital_status': tf.tensor1d(
 *       [
 *         'Never-married', 'Married-civ-spouse', 'Married-civ-spouse',
 *         'Married-civ-spouse'
 *       ],
 *       'string'),
 *   'occupation': tf.tensor1d(
 *      ['Adm-clerical', 'Craft-repair', 'Prof-specialty', 'Farming-fishing'],
 *       'string'),
 *   'relationship': tf.tensor1d(
 *       ['Not-in-family', 'Husband', 'Husband', 'Husband'], 'string'),
 *   'race': tf.tensor1d(
 *       ['White', 'Asian-Pac-Islander', 'White', 'Black'], 'string'),
 *   'sex': tf.tensor1d(['Male', 'Male', 'Male', 'Male'], 'string'),
 *   'capital_gain': tf.tensor1d([2174, 0, 0, 0], 'int32'),
 *   'capital_loss': tf.tensor1d([0, 0, 0, 0], 'int32'),
 *   'hours_per_week': tf.tensor1d([40, 40, 60, 40], 'int32'),
 *   'native_country': tf.tensor1d(
 *       ['United-States', '', 'United-States', 'United-States'], 'string')
 * };
 * const densePredictions = await tfdfModel.executeAsync(inputs);
 * console.log(densePredictions);
 * ```
 */
export class TFDFModel implements InferenceModel {
  constructor(
      private readonly graphModel: GraphModel|GraphModel<io.IOHandlerSync>,
      private readonly assets: string|Blob) {}

  get inputs() {
    return this.graphModel.inputs;
  }

  get outputs() {
    return this.graphModel.outputs;
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
   */
  predict(inputs: Tensor|Tensor[]|NamedTensorMap, config?: ModelPredictConfig):
      Tensor|Tensor[]|NamedTensorMap {
    setAssets(this.assets);
    return this.graphModel.predict(inputs, config);
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
    setAssets(this.assets);
    return this.graphModel.execute(inputs, outputs);
  }

  /**
   * Executes inference for the model for given input tensors in async
   * fashion, use this method when your model contains control flow ops.
   * @param inputs tensor, tensor array or tensor map of the inputs for the
   * model, keyed by the input node names.
   * @param outputs output node name from the TensorFlow model, if no outputs
   * are specified, the default outputs of the model would be used. You can
   * inspect intermediate nodes of the model by adding them to the outputs
   * array.
   *
   * @returns A Promise of single tensor if provided with a single output or
   * no outputs are provided and there is only one default output, otherwise
   * return a tensor map.
   */
  executeAsync(
      inputs: Tensor|Tensor[]|NamedTensorMap,
      outputs?: string|string[]): Promise<Tensor|Tensor[]> {
    setAssets(this.assets);
    return this.graphModel.executeAsync(inputs, outputs);
  }

  dispose() {
    this.graphModel.dispose();
  }
}

/**
 * Load a TFDF graph model given a URL to the model definition.
 *
 * Example of loading an example model from a URL and making a prediction with
 * an input map:
 *
 * ```js
 * // Load the test TFDF model.
 * const tfdfModel = await tfdf.loadTFDFModel(
 *     'https://storage.googleapis.com/tfjs-testing/adult_gbt_no_regex/model.json');
 * ```
 * @param modelUrl The url or an `TFDFLoadHandler` that loads the model and
 *    model assets.
 * @param options Options for the HTTP request, which allows to send credentials
 *    and custom headers.
 */
export async function loadTFDFModel(
    modelUrl: string|TFDFLoadHandler, options: io.LoadOptions = {},
    tfio = io): Promise<TFDFModel> {
  if (modelUrl == null) {
    throw new Error(
        'modelUrl in loadTFDFModel() cannot be null. Please provide a ' +
        'url or an TFDFLoadHandler that loads the model');
  }
  let graphModelUrl: Parameters<typeof loadGraphModel>[0];
  let assets: string|Blob;

  if (typeof modelUrl === 'string') {
    graphModelUrl = modelUrl;
    assets = new URL('assets.zip', modelUrl).href;
  } else {
    graphModelUrl = {load: modelUrl.loadModel};
    assets = await modelUrl.loadAssets();
  }

  const graphModel = await loadGraphModel(graphModelUrl, options, tfio);
  return new TFDFModel(graphModel, assets);
}

/**
 * Load a TFDF graph model given a synchronous IO handler with a 'load' method.
 *
 * @param modelSource The `TFDFLoadHandlerSync` that loads the model.
 */

export function loadTFDFModelSync(modelSource: TFDFLoadHandlerSync): TFDFModel {
  if (modelSource == null) {
    throw new Error(
        'modelUrl in loadTFDFModelSync() cannot be null. Please provide a ' +
        'url or an TFDFLoadHandlerSync that loads the model');
  }
  if (!modelSource.loadModel) {
    throw new Error(
        `modelUrl IO Handler ${modelSource} has no loadModel function`);
  }
  if (!modelSource.loadAssets) {
    throw new Error(
        `modelUrl IO Handler ${modelSource} has no loadAssets function`);
  }

  const graphModelSource = {load: modelSource.loadModel};
  const assets = modelSource.loadAssets();
  const graphModel = loadGraphModelSync(graphModelSource);
  return new TFDFModel(graphModel, assets);
}
