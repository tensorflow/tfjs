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

import * as tf from '@tensorflow/tfjs';
import {NamedTensorMap, TFModel} from '@tensorflow/tfjs-converter';
import {IMAGENET_CLASSES} from './imagenet_classes';

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/tfjs-models/savedmodel/';
const MODEL_FILE_URL = 'mobilenet_v1_1.0_224/optimized_model.pb';
const WEIGHT_MANIFEST_FILE_URL = 'mobilenet_v1_1.0_224/weights_manifest.json';
const INPUT_NODE_NAME = 'input';
const OUTPUT_NODE_NAME = 'MobilenetV1/Predictions/Reshape_1';
const PREPROCESS_DIVISOR = tf.scalar(255 / 2);

export class MobileNet {
  // yolo variables

  constructor() {
    this.model = new TFModel(
      GOOGLE_CLOUD_STORAGE_DIR + MODEL_FILE_URL,
      GOOGLE_CLOUD_STORAGE_DIR + WEIGHT_MANIFEST_FILE_URL);
  }

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
  predict(input) {
    const preprocessedInput = tf.div(
        tf.sub(input.asType('float32'), PREPROCESS_DIVISOR),
        PREPROCESS_DIVISOR);
    const reshapedInput =
        preprocessedInput.reshape([1, ...preprocessedInput.shape]);
    const dict = {};
    dict[INPUT_NODE_NAME] = reshapedInput;
    return this.model.eval(dict)[OUTPUT_NODE_NAME];
  }

  getTopKClasses(predictions, topK) {
    const values = predictions.dataSync();
    predictions.dispose();

    let predictionList = [];
    for (let i = 0; i < values.length; i++) {
      predictionList.push({value: values[i], index: i});
    }
    predictionList = predictionList.sort((a, b) => {
      return b.value - a.value;
    }).slice(0, topK);

    return predictionList.map(x => {
      return {label: IMAGENET_CLASSES[x.index], value: x.value};
    });
  }
}
