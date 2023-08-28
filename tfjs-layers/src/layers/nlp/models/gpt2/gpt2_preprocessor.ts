/**
 * @license
 * Copyright 2023 Google LLC.
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

/**
 * GPT-2 preprocessor layer.
 */

/* Original source: keras-nlp/models/gpt2/gpt2_preprocessor.py */
import { NamedTensorMap, Tensor, Tensor2D, serialization, tidy } from '@tensorflow/tfjs-core';

import { LayerArgs } from '../../../../engine/topology';
import { Preprocessor } from '../preprocessor';
import { GPT2Tokenizer } from './gpt2_tokenizer';
import { StartEndPacker } from '../../preprocessing/start_end_packer';
import { ValueError } from '../../../../errors';

export declare interface GPT2PreprocessorArgs extends LayerArgs {
  /**
   * A GPT2Tokenizer instance.
   */
  tokenizer: GPT2Tokenizer;

  /**
   * The length of the packed inputs.
   * Defaults to 1024.
   */
  sequenceLength?: number;

  /**
   * If `true`, the preprocessor will prepend the tokenizer start token to each
   * input sequence.
   * Defaults to `true`.
   */
  addStartToken?: boolean;

  /**
   * If `true`, the preprocessor will prepend the tokenizer end token to each
   * input sequence.
   * Defaults to `true`.
   */
  addEndToken?: boolean;
}

export declare interface GPT2PreprocessorOptions {
  /**
   * Any label data. Will be passed through unaltered.
   */
  y?: Tensor;

  /**
   * Any label weight data. Will be passed through unaltered.
   */
  sampleWeight?: Tensor;

  /**
   * Pass to override the configured `sequenceLength` of the layer.
   */
  sequenceLength?: number;
}

export function packXYSampleWeight(
  x: NamedTensorMap, y?: Tensor, sampleWeight?: Tensor):
  NamedTensorMap
  | [NamedTensorMap, Tensor]
  | [NamedTensorMap, Tensor, Tensor] {

  if (y === undefined) {
    return x;
  } else if (sampleWeight === undefined) {
    return [x, y];
  } else {
    return [x, y, sampleWeight];
  }
}

/**
 * GPT2 preprocessing layer which tokenizes and packs inputs.
 *
 * This preprocessing layer will do 2 things:
 *
 * - Tokenize the inputs using the `tokenizer`.
 * - Construct a dictionary with keys `"tokenIds"`, `"paddingMask"`, that can
 *     be passed directly to a `GPT2Backbone`.
 *
 * The call method of this layer accepts three arguments, `x`, `y`, and
 * `sampleWeight`. `x` can be a string or tensor representing a single
 * segment, a list of strings representing a batch of single segments,
 * or a list of tensors representing multiple segments to be packed together.
 * `y` and `sampleWeight` are both optional, can have any format, and will be
 * passed through unaltered.
 *
 * `GPT2Preprocessor` forces the input to have only one segment, as GPT2 is
 * mainly used for generation tasks. For tasks having multi-segment inputs
 * like "glue/mnli", please use a model designed for classification purposes
 * such as BERT or RoBERTa.
 *
 * Examples:
 *
 * Directly calling the layer on data.
 * ```js
 * const features =  ['a quick fox.', 'a fox quick.'];
 * const vocabulary =
 *    new Map([['<|endoftext|>', 0], ['a', 4], ['Ġquick', 5], ['Ġfox', 6]]);
 * const merges =
 *    ['Ġ q', 'u i', 'c k', 'ui ck', 'Ġq uick', 'Ġ f', 'o x', 'Ġf ox'];
 * const tokenizer = GPT2Tokenizer({vocabulary, merges});
 *
 * const preprocessor = GPT2Preprocessor({tokenizer});
 * preprocessor.call(tensor(['the quick brown fox jumped.']))[0].print();
 * ```
 */
export class GPT2Preprocessor extends Preprocessor {
  /** @nocollapse */
  static override className = 'GPT2Preprocessor';

  protected readonly sequenceLength: number;
  protected readonly addStartToken: boolean;
  protected readonly addEndToken: boolean;
  protected readonly packer: StartEndPacker;

  constructor(args: GPT2PreprocessorArgs) {
    super(args);
    this.tokenizer = args.tokenizer;
    this.sequenceLength = args.sequenceLength ?? 1024;
    this.addStartToken = args.addStartToken ?? true;
    this.addEndToken = args.addEndToken ?? true;

    const gpt2Tokenizer = this.tokenizer as GPT2Tokenizer;
    this.packer = new StartEndPacker({
      startValue: gpt2Tokenizer.startTokenId,
      endValue: gpt2Tokenizer.endTokenId,
      padValue: gpt2Tokenizer.padTokenId,
      sequenceLength: this.sequenceLength,
    });
  }

  override getConfig(): serialization.ConfigDict {
    const config = {
      sequenceLength: this.sequenceLength,
      addStartToken: this.addStartToken,
      addEndToken: this.addEndToken,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  override call(
    inputs: Tensor|Tensor[], kwargs: GPT2PreprocessorOptions): Tensor|Tensor[] {
    return this.callAndReturnPaddingMask(inputs, kwargs).tokenIds;
  }

  private callAndReturnPaddingMask(
    inputs: Tensor|Tensor[],
    kwargs: GPT2PreprocessorOptions
  ): NamedTensorMap {
    return tidy(() => {
      if (inputs instanceof Array) {
        if (inputs.length !== 1) {
          throw new ValueError(
            'GPT2 requires each input feature to contain only ' +
            `one segment, but received ${inputs.length}. If you are using ` +
            'GPT2 for a multi-segment classification task, please refer to ' +
            'classification models like BERT or RoBERTa.'
          );
        }
        inputs = inputs[0];
      }

      const sequenceLength = kwargs.sequenceLength ?? this.sequenceLength;
      const [tokenIds, paddingMask] = this.packer.callAndReturnPaddingMask(
        this.tokenizer.call(inputs),
        {
          sequenceLength,
          addStartValue: this.addStartToken,
          addEndValue: this.addEndToken
        }
      );

      return {
        tokenIds: tokenIds as Tensor2D,
        paddingMask: paddingMask as Tensor2D
      };
    });
  }

  /**
   * Calls the layer and returns extra information like the paddingMask used to
   * pack the sequence, the label data, and the sample weights used.
   */
  callAndPackArgs(inputs: Tensor|Tensor[], kwargs: GPT2PreprocessorOptions):
    NamedTensorMap
    | [NamedTensorMap, Tensor]
    | [NamedTensorMap, Tensor, Tensor] {
    const x = this.callAndReturnPaddingMask(inputs, kwargs);
    return packXYSampleWeight(x, kwargs.y, kwargs.sampleWeight);
  }

  static override tokenizerCls<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>) {
    return GPT2Tokenizer;
  }
}
serialization.registerClass(GPT2Preprocessor);
