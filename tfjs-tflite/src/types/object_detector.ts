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

import {BaseTaskLibrary, Class} from './common';

/** ObjectDetectorOptions proto instance. */
export declare interface ObjectDetectorOptions {
  getMaxResults(): number;
  setMaxResults(value: number): ObjectDetectorOptions;
  hasMaxResults(): boolean;

  getScoreThreshold(): number;
  setScoreThreshold(value: number): ObjectDetectorOptions;
  hasScoreThreshold(): boolean;

  getNumThreads(): number;
  setNumThreads(value: number): ObjectDetectorOptions;
  hasNumThreads(): boolean;

  // TODO: add more fields as needed.
}

/** DetectionResult proto instance. */
export declare interface DetectionResult {
  getDetectionsList(): Detection[];
}

/** Detection proto instance. */
export declare interface Detection {
  getBoundingBox(): BoundingBox|null;
  getClassesList(): Class[];
}

/** BoundingBox proto instance. */
export declare interface BoundingBox {
  getHeight(): number;
  getOriginX(): number;
  getOriginY(): number;
  getWidth(): number;
}

/** ObjectDetector class type. */
export declare interface ObjectDetectorClass {
  /**
   * The factory function to create an ImageClassifier instance.
   *
   * @param model The path to load the TFLite model from, or the model content
   *     in memory.
   * @param options Available options.
   */
  create(model: string|ArrayBuffer, options?: ObjectDetectorOptions):
      Promise<ObjectDetector>;
}

/** The main ObjectDetector class interface. */
export declare class ObjectDetector extends BaseTaskLibrary {
  /** Performs detection on the given image-like element. */
  detect(input: ImageData|HTMLImageElement|HTMLCanvasElement|
         HTMLVideoElement): DetectionResult|undefined;
}
