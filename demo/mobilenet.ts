/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import * as dl from 'deeplearn';
import {Model, TensorMap} from 'tfjs-converter';
import {IMAGENET_CLASSES} from './imagenet_classes';

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/learnjs-data/tf_model_zoo/';
const MODEL_FILE_URL = 'mobilenet_v1_1.0_224/optimized_model.pb';
const WEIGHT_FILE_URL = 'mobilenet_v1_1.0_224/optimized_model.pb.weight';
const INPUT_NODE_NAME = 'input';
const OUPUT_NODE_NAME = 'MobilenetV1/Predictions/Reshape_1';

export class MobileNet {
  // yolo variables
  private PREPROCESS_DIVISOR = dl.Scalar.new(255.0 / 2);
  private model = new Model(
      GOOGLE_CLOUD_STORAGE_DIR + MODEL_FILE_URL,
      GOOGLE_CLOUD_STORAGE_DIR + WEIGHT_FILE_URL);
  constructor() {}

  async load() {
    await this.model.load();
  }

  dispose() {
    this.model.dispose();
  }
  /**
   * Infer through SqueezeNet, assumes variables have been loaded. This does
   * standard ImageNet pre-processing before inferring through the model. This
   * method returns named activations as well as pre-softmax logits.
   *
   * @param input un-preprocessed input Array.
   * @return The pre-softmax logits.
   */
  predict(input: dl.Tensor): dl.Tensor {
    const preprocessedInput = dl.div(
        dl.sub(input.asType('float32'), this.PREPROCESS_DIVISOR),
        this.PREPROCESS_DIVISOR);
    const reshapedInput =
        preprocessedInput.reshape([1, ...preprocessedInput.shape]);
    const dict: TensorMap = {};
    dict[INPUT_NODE_NAME] = reshapedInput;
    return this.model.predict(dict)[OUPUT_NODE_NAME];
  }

  async getTopKClasses(predictions: dl.Tensor1D, topK: number, offset = 0):
      Promise<{[className: string]: number}> {
    const topk = this.topK(predictions.dataSync() as Float32Array, topK);
    predictions.dispose();
    const topkIndices = topk.indices;
    const topkValues = topk.values;

    const topClassesToProbability: {[className: string]: number} = {};
    for (let i = 0; i < topkIndices.length; i++) {
      topClassesToProbability[IMAGENET_CLASSES[topkIndices[i] + offset]] =
          topkValues[i];
    }
    return topClassesToProbability;
  }

  private topK(values: Float32Array, k: number):
      {values: Float32Array, indices: Int32Array} {
    const valuesAndIndices: Array<{value: number, index: number}> = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });
    const topkValues = new Float32Array(k);
    const topkIndices = new Int32Array(k);
    for (let i = 0; i < k; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }
    return {values: topkValues, indices: topkIndices};
  }
}
