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

import {GraphModel, loadGraphModel, loadGraphModelSync, registerOp} from '@tensorflow/tfjs-converter';
import {InferenceModel, io, ModelPredictConfig, NamedTensorMap, Tensor, tensor1d} from '@tensorflow/tfjs-core';

import * as tfdfWebAPIClient from './tfdf_web_api_client';
import {TFDFLoadHandler, TFDFLoadHandlerSync} from './types/tfdf_io';
import {TFDFWebModelRunner} from './types/tfdf_web_model_runner';

/**
 * To load a `tfdf.TFDFModel`, use the `loadTFDFModel` function below.
 *
 * Sample usage:
 *
 * ```js
 * // Load the MobilenetV2 tfdf model from tfhub.
 * const tfliteModel = await tfdf.loadTFDFModel(
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
export class TFDFModel implements InferenceModel {
  private modelRunner: TFDFWebModelRunner;
  constructor(
      private readonly graphModel: GraphModel|GraphModel<io.IOHandlerSync>,
      private readonly assets: string|Blob) {}

  get inputs() {
    return this.graphModel.inputs;
  }

  get outputs() {
    return this.graphModel.outputs;
  }

  private registerTFDFOps() {
    registerOp('SimpleMLCreateModelResource', () => [null]);
    registerOp('SimpleMLLoadModelFromPathWithHandle', async () => {
      const tfweb = await tfdfWebAPIClient.tfweb;
      this.modelRunner = await tfweb.loadModelFromUrl(this.assets);
      return [];
    });
    registerOp('SimpleMLInferenceOpWithHandle', (node) => {
      const inputs = node.inputs.map(input => input.dataSync());
      const densePredictions = this.modelRunner.predict(
          inputs[0] as Float32Array, inputs[1] as Float32Array,
          inputs[2] as Int32Array, inputs[3] as Int32Array,
          inputs[4] as Int32Array, inputs[5] as Int32Array);
      const densePredictionsTensor = tensor1d(densePredictions, 'float32');
      return [densePredictionsTensor];
    });
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
    this.registerTFDFOps();
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
    this.registerTFDFOps();
    return this.graphModel.execute(inputs, outputs);
  }
}

/**
 * Load a TFDF graph model given a URL to the model definition.
 *
 * Example of loading MobileNetV2 from a URL and making a prediction with a
 * zeros input:
 *
 * ```js
 * const modelUrl =
 *    'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';
 * const model = await tf.loadGraphModel(modelUrl);
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * model.predict(zeros).print();
 * ```
 *
 * Example of loading MobileNetV2 from a TF Hub URL and making a prediction
 * with a zeros input:
 *
 * ```js
 * const modelUrl =
 *    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
 * const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * model.predict(zeros).print();
 * ```
 * @param modelUrl The url or an `io.IOHandler` that loads the model.
 * @param options Options for the HTTP request, which allows to send
 *     credentials
 *    and custom headers.
 *
 * @doc {heading: 'Models', subheading: 'Loading'}
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
    assets = new URL('../assets', modelUrl).href;
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
 * @param modelSource The `io.IOHandlerSync` that loads the model.
 *
 * @doc {heading: 'Models', subheading: 'Loading'}
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
