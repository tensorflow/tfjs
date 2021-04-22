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

import * as tfliteWebAPIClient from '../tflite_web_api_client';
import {ImageClassifier as ImageClassifierInstance} from '../types/image_classifier';

/** ImageClassifier options. */
export interface ImageClassifierOptions {
  /**
   * Maximum number of top scored results to return. If < 0, all results will
   * be returned. If 0, an invalid argument error is returned.
   */
  maxResults?: number;

  /**
   * Score threshold in [0,1), overrides the ones provided in the model metadata
   * (if any). Results below this value are rejected.
   */
  scoreThreshold?: number;

  /**
   * The number of threads to be used for TFLite ops that support
   * multi-threading when running inference with CPU. num_threads should be
   * greater than 0 or equal to -1. Setting num_threads to -1 has the effect to
   * let TFLite runtime set the value.
   *
   * Default to -1.
   */
  numThreads?: number;
}

/** A single class in the classification result. */
export interface Class {
  className: string;
  probability: number;
}

/**
 * Client for ImageClassifier TFLite Task Library.
 *
 * It is a wrapper around the underlying javascript API to make it more
 * convenient to use. See comments in the corresponding type declaration file in
 * src/types for more info.
 */
export class ImageClassifier {
  constructor(private instance: ImageClassifierInstance) {}

  static async create(
      model: string|ArrayBuffer,
      options?: ImageClassifierOptions): Promise<ImageClassifier> {
    const optionsProto = new tfliteWebAPIClient.tfweb.ImageClassifierOptions();
    if (options) {
      if (options.maxResults !== undefined) {
        optionsProto.setMaxResults(options.maxResults);
      }
      if (options.scoreThreshold !== undefined) {
        optionsProto.setScoreThreshold(options.scoreThreshold);
      }
      if (options.numThreads !== undefined) {
        optionsProto.setNumThreads(options.numThreads);
      }
    }
    const instance = await tfliteWebAPIClient.tfweb.ImageClassifier.create(
        model, optionsProto);
    return new ImageClassifier(instance);
  }

  classify(input: ImageData|HTMLImageElement|HTMLCanvasElement|
           HTMLVideoElement): Class[] {
    const result = this.instance.classify(input);
    if (!result) {
      return [];
    }

    const classes: Class[] = [];
    if (result.getClassificationsList().length > 0) {
      result.getClassificationsList()[0].getClassesList().forEach(cls => {
        classes.push({
          className: cls.getDisplayName() || cls.getClassName(),
          probability: cls.getScore(),
        });
      });
    }
    return classes;
  }

  cleanUp() {
    this.instance.cleanUp();
  }
}
