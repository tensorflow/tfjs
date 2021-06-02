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

/** ImageClassifierOptions proto instance. */
export declare interface ImageClassifierOptions {
  getMaxResults(): number;
  setMaxResults(value: number): ImageClassifierOptions;
  hasMaxResults(): boolean;

  getScoreThreshold(): number;
  setScoreThreshold(value: number): ImageClassifierOptions;
  hasScoreThreshold(): boolean;

  getNumThreads(): number;
  setNumThreads(value: number): ImageClassifierOptions;
  hasNumThreads(): boolean;

  // TODO: add more fields as needed.
}

/** ClassificationResult proto instance. */
export declare interface ClassificationResult {
  getClassificationsList(): Classification[];
}

/** Classification proto instance. */
export declare interface Classification {
  getClassesList(): Class[];
}

/** ImageClassifier class type. */
export declare interface ImageClassifierClass {
  /**
   * The factory function to create an ImageClassifier instance.
   *
   * @param model The path to load the TFLite model from, or the model content
   *     in memory.
   * @param options Available options.
   */
  create(model: string|ArrayBuffer, options?: ImageClassifierOptions):
      Promise<ImageClassifier>;
}

/** The main ImageClassifier class interface. */
export declare class ImageClassifier extends BaseTaskLibrary {
  /** Performs classification on the given image-like element. */
  classify(input: ImageData|HTMLImageElement|HTMLCanvasElement|
           HTMLVideoElement): ClassificationResult|undefined;
}
