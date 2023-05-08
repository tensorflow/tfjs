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

import {BaseTaskLibrary} from './common';

/** A single answer. */
export declare interface QaAnswer {
  text: string;
  pos: Pos;
}

/** Answer position. */
export declare interface Pos {
  start: number;
  end: number;
  logit: number;
}

/** BertQuestionAnswerer class type. */
export declare interface BertQuestionAnswererClass {
  /**
   * The factory function to create an ImageClassifier instance.
   *
   * @param model The path to load the TFLite model from, or the model content
   *     in memory.
   */
  create(model: string|ArrayBuffer): Promise<BertQuestionAnswerer>;
}

/** The main BertQuestionAnswerer class instance. */
export declare class BertQuestionAnswerer extends BaseTaskLibrary {
  /** Answers question based on the context. */
  answer(context: string, question: string): QaAnswer[]|undefined;
}
