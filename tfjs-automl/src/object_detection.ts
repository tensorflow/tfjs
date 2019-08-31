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
import {dispose, getBackend, image, setBackend, Tensor, Tensor2D, tidy} from '@tensorflow/tfjs-core';

import {ImageInput} from './types';
import {imageToTensor, loadDictionary} from './util';

const DEFAULT_TOPK = 20;
const DEFAULT_IOU_THRESHOLD = 0.5;
const DEFAULT_SCORE_THRESHOLD = 0.5;

const INPUT_NODE_NAME = 'ToFloat';
const OUTPUT_NODE_NAMES =
    ['Postprocessor/convert_scores', 'Postprocessor/Decode/transpose_1'];

export interface ObjectDetectionOptions {
  topk: number;
  iou: number;
  score: number;
}

export interface Box {
  top: number;
  left: number;
  width: number;
  height: number;
}
export interface PredictedObject {
  box: Box;
  score: number;
  label: string;
}
export type ObjectDetectionPrediction = PredictedObject[];

export class ObjectDetectionModel {
  constructor(public graphModel: GraphModel, public dictionary: string[]) {}

  async detect(input: ImageInput, options?: ObjectDetectionOptions):
      Promise<ObjectDetectionPrediction> {
    options = sanitizeOptions(options);
    const img = tidy(() => this.preprocess(input, options));
    const [height, width] = [img.shape[1], img.shape[2]];
    const feedDict: {[name: string]: Tensor} = {};
    feedDict[INPUT_NODE_NAME] = img;
    const [scoresTensor, boxesTensor] =
        await this.graphModel.executeAsync(feedDict, OUTPUT_NODE_NAMES) as
        Tensor[];

    const [, numBoxes, numClasses] = scoresTensor.shape;
    const [scores, boxes] =
        await Promise.all([scoresTensor.data(), boxesTensor.data()]);
    const {maxScores, classes} =
        calculateMaxScores(scores as Float32Array, numBoxes, numClasses);

    // Run post process in cpu for speed.
    const prevBackend = getBackend();
    setBackend('cpu');
    // Sort the boxes by score, ignoring overlapping boxes.
    const topIndicesTensor = image.nonMaxSuppression(
        boxesTensor as Tensor2D, maxScores, options.topk, options.iou,
        options.score);
    const topIndices = topIndicesTensor.dataSync() as Int32Array;
    dispose([img, scoresTensor, boxesTensor, topIndicesTensor]);
    // Restore the previous backend.
    setBackend(prevBackend);
    const result = buildDetectedObjects(
        width, height, boxes as Float32Array, maxScores, topIndices, classes,
        this.dictionary);
    return result;
  }

  private preprocess(input: ImageInput, options: ObjectDetectionOptions) {
    return imageToTensor(input).expandDims().toFloat();
  }
}

export async function loadObjectDetectionModel(modelUrl: string):
    Promise<ObjectDetectionModel> {
  const [model, dict] =
      await Promise.all([loadGraphModel(modelUrl), loadDictionary(modelUrl)]);
  return new ObjectDetectionModel(model, dict);
}

function sanitizeOptions(options: ObjectDetectionOptions) {
  options = options || {} as ObjectDetectionOptions;
  if (options.topk == null) {
    options.topk = DEFAULT_TOPK;
  }
  if (options.iou == null) {
    options.iou = DEFAULT_IOU_THRESHOLD;
  }
  if (options.score == null) {
    options.score = DEFAULT_SCORE_THRESHOLD;
  }
  return options;
}

function calculateMaxScores(
    scores: Float32Array, numBoxes: number,
    numClasses: number): {maxScores: number[], classes: number[]} {
  const maxScores = [];
  const classes: number[] = [];
  for (let i = 0; i < numBoxes; i++) {
    let max = Number.MIN_VALUE;
    let classIndex = -1;
    for (let j = 0; j < numClasses; j++) {
      const score = scores[i * numClasses + j];
      if (score > max) {
        max = scores[i * numClasses + j];
        classIndex = j;
      }
    }
    maxScores[i] = max;
    classes[i] = classIndex;
  }
  return {maxScores, classes};
}

function buildDetectedObjects(
    width: number, height: number, boxes: Float32Array, scores: number[],
    indexes: Int32Array, classes: number[],
    dictionary: string[]): ObjectDetectionPrediction {
  const count = indexes.length;
  const objects = [];
  for (let i = 0; i < count; i++) {
    const bbox = [];
    for (let j = 0; j < 4; j++) {
      bbox[j] = boxes[indexes[i] * 4 + j];
    }
    const minY = bbox[0] * height;
    const minX = bbox[1] * width;
    const maxY = bbox[2] * height;
    const maxX = bbox[3] * width;
    bbox[0] = minX;
    bbox[1] = minY;
    bbox[2] = maxX - minX;
    bbox[3] = maxY - minY;
    objects.push({
      box: {
        left: minX,
        top: minY,
        width: maxX - minX,
        height: maxY - minY,
      },
      label: dictionary[classes[indexes[i]]],
      score: scores[indexes[i]],
    });
  }
  return objects;
}
