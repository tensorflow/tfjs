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

export interface ClassificationPrediction {
  label: string;
  probabilities: number[];
}

export interface ImageClassificationOptions {
  centerCrop: boolean;
}

const IMG_SIZE: [number, number] = [224, 224];

export class ImageClassificationModel {
  constructor(private model: GraphModel) {}

  async predict(input: ImageInput, options?: ImageClassificationOptions):
      Promise<ClassificationPrediction> {
    options = options || {} as ImageClassificationOptions;
    if (options.centerCrop == null) {
      options.centerCrop = true;
    }
    // Preprocessing involves center crop and normalizing between [-1, 1].
    const scores = tidy(() => {
      const img = imageToTensor(input);
      const croppedImg = options.centerCrop ?
          centerCrop(img) :
          image.resizeBilinear(img, IMG_SIZE);
      const normalizedImg = croppedImg.div(127.5).sub(1);
      return this.model.predict(normalizedImg) as Tensor;
    });
    const probabilities = await scores.data();

    return {
      probabilities, label
    }
  }
}

export type ImageInput =
    ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|Tensor3D;

export async function loadImageClassification(modelUrl: string):
    Promise<ImageClassificationModel> {
  const model = await loadGraphModel(modelUrl);
  return new ImageClassificationModel(model);
}

export function imageToTensor(img: ImageInput): Tensor3D {
  return img instanceof Tensor ? img : browser.fromPixels(img);
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
