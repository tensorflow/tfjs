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
 * Unit Tests for Preprocessor Layers.
 */
import { BytePairTokenizer } from '../tokenizers';
import { Preprocessor } from './preprocessor';

describe('Preprocessor', () => {
  let preprocessor: Preprocessor;

  beforeEach(() => {
    preprocessor = new Preprocessor({});
  });

  it('serialization round-trip with no set tokenizer', () => {
    const reserialized = Preprocessor.fromConfig(
      Preprocessor, preprocessor.getConfig());
    expect(reserialized.getConfig()).toEqual(preprocessor.getConfig());
  });

  it('serialization round-trip with set tokenizer', () => {
    preprocessor.tokenizer = new BytePairTokenizer({
      vocabulary: new Map([['<|endoftext|>', 0]]), merges: ['a b']});

    const reserialized = Preprocessor.fromConfig(
      Preprocessor, preprocessor.getConfig());
    expect(reserialized.getConfig()).toEqual(preprocessor.getConfig());
  });
});
