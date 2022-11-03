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
import {NLClassifier as TaskLibraryNLClassifier} from '../types/nl_classifier';
import {BaseTaskLibraryClient, Class} from './common';

/**
 * NLClassifier options.
 */
export declare interface NLClassifierOptions {
  /** Index of the input tensor. */
  inputTensorIndex: number;
  /** Index of the output score tensor. */
  outputScoreTensorIndex: number;
  /** Index of the output label tensor. */
  outputLabelTensorIndex: number;
  /** Name of the input tensor. */
  inputTensorName: string;
  /** Name of the output score tensor. */
  outputScoreTensorName: string;
  /** Name of the output label tensor. */
  outputLabelTensorName: string;
}

/**
 * Default NLClassifier options.
 */
const DEFAULT_NLCLASSIFIER_OPTIONS: NLClassifierOptions = {
  inputTensorIndex: 0,
  outputScoreTensorIndex: 0,
  outputLabelTensorIndex: -1,
  inputTensorName: 'INPUT',
  outputScoreTensorName: 'OUTPUT_SCORE',
  outputLabelTensorName: 'OUTPUT_LABEL',
};

/**
 * Client for NLClassifier TFLite Task Library.
 *
 * It is a wrapper around the underlying javascript API to make it more
 * convenient to use. See comments in the corresponding type declaration file in
 * src/types for more info.
 */
export class NLClassifier extends BaseTaskLibraryClient {
  constructor(protected override instance: TaskLibraryNLClassifier) {
    super(instance);
  }

  static async create(
      model: string|ArrayBuffer,
      options = DEFAULT_NLCLASSIFIER_OPTIONS): Promise<NLClassifier> {
    const instance =
        await tfliteWebAPIClient.tfweb.NLClassifier.create(model, options);
    return new NLClassifier(instance);
  }

  classify(input: string): Class[]|undefined {
    return this.instance.classify(input).map(category => {
      return {
        probability: category.score,
        className: category.className,
      };
    });
  }
}
