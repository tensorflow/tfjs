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
import { GraphModel } from '@tensorflow/tfjs-converter';
import { ImageInput } from './types';
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
export declare class ObjectDetectionModel {
    graphModel: GraphModel;
    dictionary: string[];
    constructor(graphModel: GraphModel, dictionary: string[]);
    detect(input: ImageInput, options?: ObjectDetectionOptions): Promise<PredictedObject[]>;
    private preprocess;
}
export declare function loadObjectDetection(modelUrl: string): Promise<ObjectDetectionModel>;
