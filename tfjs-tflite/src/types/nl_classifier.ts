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

import {BaseTaskLibrary, Category} from './common';

/** Classification options to identify input and ouptut tensors of the model. */
export declare interface NLClassifierOptions {
  inputTensorIndex: number;
  outputScoreTensorIndex: number;
  outputLabelTensorIndex: number;
  inputTensorName: string;
  outputScoreTensorName: string;
  outputLabelTensorName: string;
}

/** NLClassifier class type. */
export declare interface NLClassifierClass {
  /**
   * The factory function to create a NLClassifier instance.
   *
   * @param model The path to load the TFLite model from, or the model content
   *     in memory.
   * @param options Available options.
   */
  create(model: string|ArrayBuffer, options: NLClassifierOptions):
      Promise<NLClassifier>;
}

/** The main NLClassifier class interface. */
export declare interface NLClassifier extends BaseTaskLibrary {
  /** Performs classification on a string input, returns classified results. */
  classify(input: string): Category[]|undefined;
}
