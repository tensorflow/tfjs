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
import {ImageClassifier as TaskLibraryImageClassifier} from '../types/image_classifier';

import {BaseTaskLibraryClient, Class, CommonTaskLibraryOptions, convertProtoClassesToClasses, getDefaultNumThreads} from './common';

/** ImageClassifier options. */
export interface ImageClassifierOptions extends CommonTaskLibraryOptions {
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
}

/**
 * Client for ImageClassifier TFLite Task Library.
 *
 * It is a wrapper around the underlying javascript API to make it more
 * convenient to use. See comments in the corresponding type declaration file in
 * src/types for more info.
 */
export class ImageClassifier extends BaseTaskLibraryClient {
  constructor(protected override instance: TaskLibraryImageClassifier) {
    super(instance);
  }

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
    if (!options || options.numThreads === undefined) {
      optionsProto.setNumThreads(await getDefaultNumThreads());
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

    let classes: Class[] = [];
    if (result.getClassificationsList().length > 0) {
      classes = convertProtoClassesToClasses(
          result.getClassificationsList()[0].getClassesList());
    }
    return classes;
  }
}
