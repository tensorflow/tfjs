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
import { Tokenizer } from '../tokenizers';
import { Kwargs } from '../../../types';
import { deserializeKerasObject, serializeKerasObject } from '../../../utils/generic_utils';

/**
 * Base class for model Preprocessors.
 */
export class Preprocessor extends Layer {
  /** @nocollapse */
  static className = 'Preprocessor';

  private _tokenizer: Tokenizer;

  constructor(args: LayerArgs) {
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
    const config = super.getConfig();
    config.tokenizer = serializeKerasObject(this.tokenizer);
    return config;
  }

  static override fromConfig<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>,
    config: serialization.ConfigDict
  ): T {
    const kwargs: Kwargs = config;

    if (config.tokenizer != null && !(config.tokenizer instanceof Tokenizer)) {
      const tokenizerConfigDict = config.tokenizer as serialization.ConfigDict;

      kwargs.tokenizer = deserializeKerasObject(
        tokenizerConfigDict,
        serialization.SerializationMap.getMap().classNameMap,
        {}, 'preprocessor');
    }
    return new cls(kwargs);
  }

  static tokenizerCls<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>) {}
}
serialization.registerClass(Preprocessor);
