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

/* Original source: keras-nlp/models/preprocessor.py */
import { serialization } from '@tensorflow/tfjs-core';

import { Layer, LayerArgs } from '../../../engine/topology';
import { Kwargs } from '../../../types';
import { Tokenizer } from '../tokenizers';
import { NotImplementedError } from 'tfjs-layers/src/errors';

export declare interface StartEndPackerArgs extends LayerArgs {
  /**
   * Integer. The desired output length.
   */
  sequenceLength: number;

  /**
   * Integer or string. The ID or token that is to be placed at the start of
   * each sequence. The dtype must match the dtype of the input tensors to the
   * layer. If undefined, no start value will be added.
   */
  startValue?: number|string;

  /**
   * Integer or string. The ID or token that is to be placed at the end of each
   * input segment. The dtype must match the dtype of the input tensors to the
   * layer. If undefined, no end value will be added.
   */
  endValue?: number|string;

  /**
   * Integer or string. The ID or token that is to be placed into the unused
   * positions after the last segment in the sequence. If undefined, 0 or ''
   * will be added depending on the dtype of the input tensor.
   */
  padValue?: number|string;
}

export declare interface StartEndPackerOptions {
  /**
   * Pass to override the configured `sequenceLength` of the layer.
   */
  sequenceLength?: number;

  /**
   * Pass `false` to not append a start value for this input.
   * Defaults to true.
   */
  addStartValue?: boolean;

  /**
   * Pass `false` to not append an end value for this input.
   * Defaults to true.
   */
  addEndValue?: boolean;
}

/**
 * Base class for model Preprocessors.
 */
export class Preprocessor extends Layer {
  /** @nocollapse */
  static readonly className = 'Preprocessor';

  private _tokenizer: Tokenizer;

  constructor(args: Kwargs) {
    super(args);
  }

  /**
   * The tokenizer used to tokenize strings.
   */
  get tokenizer() {
    return this._tokenizer;
  }

  set tokenizer(value: Tokenizer) {
    this._tokenizer = value;
  }

  override getConfig(): serialization.ConfigDict {
    const config = {
      tokenizer: this._tokenizer.getClassName(),
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  static override fromConfig<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>,
    config: serialization.ConfigDict
  ): T {
    // TODO(orderique): Find out the correct way to deserialize this.
    throw new NotImplementedError('Not implemented yet for Preprocessors.');
  }

  static tokenizerCls<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>) {}

  static presets<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>) {
    return {};
  }

  static fromPreset<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>,
    preset: string,
    kwargs: Kwargs
  ) {
    // TODO(orderique): Discuss the right way to approach this.
    throw new NotImplementedError('Not implemented yet.');
  }
}
