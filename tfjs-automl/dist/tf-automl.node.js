/**
 * @license
 * Copyright 2024 Google LLC. All Rights Reserved.
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
'use strict';

var tfjsConverter = require('@tensorflow/tfjs-converter');
var tfjsCore = require('@tensorflow/tfjs-core');

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
function imageToTensor(img) {
    return img instanceof tfjsCore.Tensor ? img : tfjsCore.browser.fromPixels(img);
}
/** Loads and parses the dictionary. */
async function loadDictionary(modelUrl) {
    const lastIndexOfSlash = modelUrl.lastIndexOf('/');
    const prefixUrl = lastIndexOfSlash >= 0 ? modelUrl.slice(0, lastIndexOfSlash + 1) : '';
    const dictUrl = `${prefixUrl}dict.txt`;
    const response = await tfjsCore.util.fetch(dictUrl);
    const text = await response.text();
    return text.trim().split('\n');
}

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
/** Input size as expected by the model. */
const IMG_SIZE = [224, 224];
// Constants used to normalize the image between -1 and 1.
const DIV_FACTOR = 127.5;
const SUB_FACTOR = 1;
class ImageClassificationModel {
    constructor(graphModel, dictionary) {
        this.graphModel = graphModel;
        this.dictionary = dictionary;
    }
    async classify(input, options) {
        options = sanitizeOptions$1(options);
        const scores = tfjsCore.tidy(() => {
            const preprocessedImg = this.preprocess(input, options);
            return this.graphModel.predict(preprocessedImg);
        });
        const probabilities = await scores.data();
        scores.dispose();
        const result = Array.from(probabilities)
            .map((prob, i) => ({ label: this.dictionary[i], prob }));
        return result;
    }
    preprocess(input, options) {
        // Preprocessing involves center crop and normalizing between [-1, 1].
        const img = imageToTensor(input);
        const croppedImg = options.centerCrop ?
            centerCropAndResize(img) :
            tfjsCore.expandDims(tfjsCore.image.resizeBilinear(img, IMG_SIZE));
        return tfjsCore.sub(tfjsCore.div(croppedImg, DIV_FACTOR), SUB_FACTOR);
    }
}
async function loadImageClassification(modelUrl) {
    const [model, dict] = await Promise.all([tfjsConverter.loadGraphModel(modelUrl), loadDictionary(modelUrl)]);
    return new ImageClassificationModel(model, dict);
}
function sanitizeOptions$1(options) {
    options = options || {};
    if (options.centerCrop == null) {
        options.centerCrop = true;
    }
    return options;
}
/** Center crops an image */
function centerCropAndResize(img) {
    return tfjsCore.tidy(() => {
        const [height, width] = img.shape.slice(0, 2);
        let top = 0;
        let left = 0;
        if (height > width) {
            top = (height - width) / 2;
        }
        else {
            left = (width - height) / 2;
        }
        const size = Math.min(width, height);
        const boxes = [
            [top / height, left / width, (top + size) / height, (left + size) / width]
        ];
        const boxIndices = [0];
        return tfjsCore.image.cropAndResize(
        // tslint:disable-next-line
        tfjsCore.expandDims(tfjsCore.cast(img, 'float32')), boxes, boxIndices, IMG_SIZE);
    });
}

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
const DEFAULT_TOPK = 20;
const DEFAULT_IOU_THRESHOLD = 0.5;
const DEFAULT_SCORE_THRESHOLD = 0.5;
const INPUT_NODE_NAME = 'ToFloat';
const OUTPUT_NODE_NAMES = ['Postprocessor/convert_scores', 'Postprocessor/Decode/transpose_1'];
class ObjectDetectionModel {
    constructor(graphModel, dictionary) {
        this.graphModel = graphModel;
        this.dictionary = dictionary;
    }
    async detect(input, options) {
        options = sanitizeOptions(options);
        const img = tfjsCore.tidy(() => this.preprocess(input, options));
        const [height, width] = [img.shape[1], img.shape[2]];
        const feedDict = {};
        feedDict[INPUT_NODE_NAME] = img;
        const [scoresTensor, boxesTensor] = await this.graphModel.executeAsync(feedDict, OUTPUT_NODE_NAMES);
        const [, numBoxes, numClasses] = scoresTensor.shape;
        const [scores, boxes] = await Promise.all([scoresTensor.data(), boxesTensor.data()]);
        const { boxScores, boxLabels } = calculateMostLikelyLabels(scores, numBoxes, numClasses);
        // Sort the boxes by score, ignoring overlapping boxes.
        const selectedBoxesTensor = await tfjsCore.image.nonMaxSuppressionAsync(boxesTensor, boxScores, options.topk, options.iou, options.score);
        const selectedBoxes = await selectedBoxesTensor.data();
        tfjsCore.dispose([img, scoresTensor, boxesTensor, selectedBoxesTensor]);
        const result = buildDetectedObjects(width, height, boxes, boxScores, boxLabels, selectedBoxes, this.dictionary);
        return result;
    }
    preprocess(input, options) {
        return tfjsCore.cast(tfjsCore.expandDims(imageToTensor(input)), 'float32');
    }
}
async function loadObjectDetection(modelUrl) {
    const [model, dict] = await Promise.all([tfjsConverter.loadGraphModel(modelUrl), loadDictionary(modelUrl)]);
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

/** @license See the LICENSE file. */
// This code is auto-generated, do not modify this file!
const version = '1.2.0';

exports.ImageClassificationModel = ImageClassificationModel;
exports.ObjectDetectionModel = ObjectDetectionModel;
exports.loadImageClassification = loadImageClassification;
exports.loadObjectDetection = loadObjectDetection;
exports.version = version;
//# sourceMappingURL=tf-automl.node.js.map
