/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {GraphModel, loadGraphModel} from '@tensorflow/tfjs-converter';
import {browser, image, Tensor, Tensor3D, tidy} from '@tensorflow/tfjs-core';

import {ClassificationPrediction, ImageClassificationOptions, ImageInput} from './types';

/** Input size as expected by the model. */
const IMG_SIZE: [number, number] = [224, 224];
// Constants used to normalize the image between -1 and 1.
const DIV_FACTOR = 127.5;
const SUB_FACTOR = 1;

export class ImageClassificationModel {
  constructor(public graphModel: GraphModel, public dictionary: string[]) {}

  async classify(input: ImageInput, options?: ImageClassificationOptions):
      Promise<ClassificationPrediction> {
    options = sanitizeOptions(options);

    const scores = tidy(() => {
      const preprocessedImg = this.preprocess(input, options);
      return this.graphModel.predict(preprocessedImg) as Tensor;
    });
    const probabilities = await scores.data() as Float32Array;
    scores.dispose();
    const result = Array.from(probabilities)
                       .map((prob, i) => ({label: this.dictionary[i], prob}));
    return result;
  }

  private preprocess(input: ImageInput, options: ImageClassificationOptions) {
    // Preprocessing involves center crop and normalizing between [-1, 1].
    const img = imageToTensor(input);
    const croppedImg = options.centerCrop ? centerCrop(img) :
                                            image.resizeBilinear(img, IMG_SIZE);
    return croppedImg.div(DIV_FACTOR).sub(SUB_FACTOR);
  }
}

export async function loadImageClassification(modelUrl: string):
    Promise<ImageClassificationModel> {
  const [model, dict] =
      await Promise.all([loadGraphModel(modelUrl), loadDictionary(modelUrl)]);
  return new ImageClassificationModel(model, dict);
}

function imageToTensor(img: ImageInput): Tensor3D {
  return img instanceof Tensor ? img : browser.fromPixels(img);
}

function sanitizeOptions(options: ImageClassificationOptions) {
  options = options || {} as ImageClassificationOptions;
  if (options.centerCrop == null) {
    options.centerCrop = true;
  }
  return options;
}

/** Loads and parses the dictionary. */
async function loadDictionary(modelUrl: string): Promise<string[]> {
  const prefixUrl = modelUrl.slice(0, modelUrl.lastIndexOf('/'));
  const dictUrl = `${prefixUrl}/dict.txt`;
  const response = await fetch(dictUrl);
  const text = await response.text();
  return text.split('\n');
}

/** Center crops an image */
function centerCrop(img: Tensor3D) {
  return tidy(() => {
    const [height, width] = img.shape.slice(0, 2);
    let top = 0;
    let left = 0;
    if (height > width) {
      top = (height - width) / 2;
    } else {
      left = (width - height) / 2;
    }
    const size = Math.min(width, height);
    const boxes = [
      [top / height, left / width, (top + size) / height, (left + size) / width]
    ];
    const boxIndices = [0];
    return image.cropAndResize(
        img.toFloat().expandDims(), boxes, boxIndices, IMG_SIZE);
  });
}
