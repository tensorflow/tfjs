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
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import { cast, dispose, expandDims, image, tidy } from '@tensorflow/tfjs-core';
import { imageToTensor, loadDictionary } from './util';
const DEFAULT_TOPK = 20;
const DEFAULT_IOU_THRESHOLD = 0.5;
const DEFAULT_SCORE_THRESHOLD = 0.5;
const INPUT_NODE_NAME = 'ToFloat';
const OUTPUT_NODE_NAMES = ['Postprocessor/convert_scores', 'Postprocessor/Decode/transpose_1'];
export class ObjectDetectionModel {
    constructor(graphModel, dictionary) {
        this.graphModel = graphModel;
        this.dictionary = dictionary;
    }
    async detect(input, options) {
        options = sanitizeOptions(options);
        const img = tidy(() => this.preprocess(input, options));
        const [height, width] = [img.shape[1], img.shape[2]];
        const feedDict = {};
        feedDict[INPUT_NODE_NAME] = img;
        const [scoresTensor, boxesTensor] = await this.graphModel.executeAsync(feedDict, OUTPUT_NODE_NAMES);
        const [, numBoxes, numClasses] = scoresTensor.shape;
        const [scores, boxes] = await Promise.all([scoresTensor.data(), boxesTensor.data()]);
        const { boxScores, boxLabels } = calculateMostLikelyLabels(scores, numBoxes, numClasses);
        // Sort the boxes by score, ignoring overlapping boxes.
        const selectedBoxesTensor = await image.nonMaxSuppressionAsync(boxesTensor, boxScores, options.topk, options.iou, options.score);
        const selectedBoxes = await selectedBoxesTensor.data();
        dispose([img, scoresTensor, boxesTensor, selectedBoxesTensor]);
        const result = buildDetectedObjects(width, height, boxes, boxScores, boxLabels, selectedBoxes, this.dictionary);
        return result;
    }
    preprocess(input, options) {
        return cast(expandDims(imageToTensor(input)), 'float32');
    }
}
export async function loadObjectDetection(modelUrl) {
    const [model, dict] = await Promise.all([loadGraphModel(modelUrl), loadDictionary(modelUrl)]);
    return new ObjectDetectionModel(model, dict);
}
function sanitizeOptions(options) {
    options = options || {};
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
function calculateMostLikelyLabels(scores, numBoxes, numClasses) {
    // Holds a score for each box.
    const boxScores = [];
    // Holds the label id for each box.
    const boxLabels = [];
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
    return { boxScores, boxLabels };
}
function buildDetectedObjects(width, height, boxes, boxScores, boxLabels, selectedBoxes, dictionary) {
    const objects = [];
    // Each 2d rectangle is fully described with 4 coordinates.
    const numBoxCoords = 4;
    for (let i = 0; i < selectedBoxes.length; i++) {
        const boxIndex = selectedBoxes[i];
        const [top, left, bottom, right] = Array.from(boxes.slice(boxIndex * numBoxCoords, boxIndex * numBoxCoords + numBoxCoords));
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
//# sourceMappingURL=object_detection.js.map