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
import {cast, dispose, expandDims, image, Tensor, Tensor2D, tidy} from '@tensorflow/tfjs-core';

import {ImageInput} from './types';
import {imageToTensor, loadDictionary} from './util';

const DEFAULT_TOPK = 20;
const DEFAULT_IOU_THRESHOLD = 0.5;
const DEFAULT_SCORE_THRESHOLD = 0.5;

const INPUT_NODE_NAME = 'ToFloat';
const OUTPUT_NODE_NAMES =
    ['Postprocessor/convert_scores', 'Postprocessor/Decode/transpose_1'];

export interface ObjectDetectionOptions {
  /**
   * Only the `topk` most likely objects are returned. The actual number of
   * objects might be less than this number.
   */
  topk?: number;
  /**
   * Intersection over union threshold. IoU is a metric between 0 and 1 used to
   * measure the overlap of two boxes. The predicted boxes will not overlap more
   * than the specified threshold.
   */
  iou?: number;
  /** Boxes with score lower than this threshold will be ignored. */
  score?: number;
}

/** Contains the coordinates of a bounding box. */
export interface Box {
  /** Number of pixels from the top of the image (top padding). */
  top: number;
  /** Number of pixels from the left of the image (left padding). */
  left: number;
  /** The width of the box. */
  width: number;
  /** The height of the box. */
  height: number;
}

/** The predicted object, which holds the score, label and bounding box. */
export interface PredictedObject {
  box: Box;
  score: number;
  label: string;
}

export class ObjectDetectionModel {
  constructor(public graphModel: GraphModel, public dictionary: string[]) {}

  async detect(input: ImageInput, options?: ObjectDetectionOptions):
      Promise<PredictedObject[]> {
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
    const {boxScores, boxLabels} =
        calculateMostLikelyLabels(scores as Float32Array, numBoxes, numClasses);

    // Sort the boxes by score, ignoring overlapping boxes.
    const selectedBoxesTensor = await image.nonMaxSuppressionAsync(
        boxesTensor as Tensor2D, boxScores, options.topk, options.iou,
        options.score);
    const selectedBoxes = await selectedBoxesTensor.data() as Int32Array;
    dispose([img, scoresTensor, boxesTensor, selectedBoxesTensor]);

    const result = buildDetectedObjects(
        width, height, boxes as Float32Array, boxScores, boxLabels,
        selectedBoxes, this.dictionary);
    return result;
  }

  private preprocess(input: ImageInput, options: ObjectDetectionOptions) {
    return cast(expandDims(imageToTensor(input)), 'float32');
  }
}

export async function loadObjectDetection(modelUrl: string):
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

function calculateMostLikelyLabels(
    scores: Float32Array, numBoxes: number,
    numClasses: number): {boxScores: number[], boxLabels: number[]} {
  // Holds a score for each box.
  const boxScores: number[] = [];
  // Holds the label id for each box.
  const boxLabels: number[] = [];
  for (let i = 0; i < numBoxes; i++) {
    let maxScore = Number.MIN_VALUE;
    let mostLikelyLabel = -1;
    for (let j = 0; j < numClasses; j++) {
      const flatIndex = i * numClasses + j;
      const score = scores[flatIndex];
      if (score > maxScore) {
        maxScore = scores[flatIndex];
        mostLikelyLabel = j;
      }
    }
    boxScores[i] = maxScore;
    boxLabels[i] = mostLikelyLabel;
  }
  return {boxScores, boxLabels};
}

function buildDetectedObjects(
    width: number, height: number, boxes: Float32Array, boxScores: number[],
    boxLabels: number[], selectedBoxes: Int32Array,
    dictionary: string[]): PredictedObject[] {
  const objects: PredictedObject[] = [];
  // Each 2d rectangle is fully described with 4 coordinates.
  const numBoxCoords = 4;
  for (let i = 0; i < selectedBoxes.length; i++) {
    const boxIndex = selectedBoxes[i];
    const [top, left, bottom, right] = Array.from(boxes.slice(
        boxIndex * numBoxCoords, boxIndex * numBoxCoords + numBoxCoords));
    objects.push({
      box: {
        left: left * width,
        top: top * height,
        width: (right - left) * width,
        height: (bottom - top) * height,
      },
      label: dictionary[boxLabels[boxIndex]],
      score: boxScores[boxIndex],
    });
  }
  return objects;
}
