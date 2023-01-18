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
import {BertQuestionAnswerer as TaskLibraryBertQuestionAnswerer} from '../types/bert_qa';
import {BaseTaskLibraryClient} from './common';

/** A single answer. */
export interface QaAnswer {
  /** The text of the answer. */
  text: string;
  /** The position and logit of the answer. */
  pos: Pos;
}

/** Answer position. */
export interface Pos {
  /** The start position. */
  start: number;
  /** The end position. */
  end: number;
  /** The logit. */
  logit: number;
}

/**
 * Client for BertQA TFLite Task Library.
 *
 * It is a wrapper around the underlying javascript API to make it more
 * convenient to use. See comments in the corresponding type declaration file in
 * src/types for more info.
 */
export class BertQuestionAnswerer extends BaseTaskLibraryClient {
  constructor(protected override instance: TaskLibraryBertQuestionAnswerer) {
    super(instance);
  }

  static async create(model: string|
                      ArrayBuffer): Promise<BertQuestionAnswerer> {
    const instance =
        await tfliteWebAPIClient.tfweb.BertQuestionAnswerer.create(model);
    return new BertQuestionAnswerer(instance);
  }

  answer(context: string, question: string): QaAnswer[] {
    const result = this.instance.answer(context, question);
    if (!result) {
      return [];
    }

    return result.map(answer => {
      return {
        text: answer.text,
        pos: answer.pos,
      };
    });
  }
}
