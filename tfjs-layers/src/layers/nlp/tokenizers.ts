/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 *  Tokenizer layers.
 */

/* Original source: keras-nlp/tokenizer.py */
import { Tensor1D, serialization, tensor1d } from '@tensorflow/tfjs-core';

import { Layer } from '../../engine/topology';
import { NotImplementedError, ValueError } from '../../errors';

export declare interface TokenizerOptions {
  mode?: 'tokenize' | 'detokenize'
}

/**
 * Base class for Tokenizers.
 *
 * Subclassers should always implement the `tokenize()` method, which will also
 * be the default when calling the layer directly on inputs.
 */
export abstract class Tokenizer extends Layer {
  /**
   * Transform input tensors of strings into output tokens.
   *
   * @param inputs Input tensor.
   * @param kwargs Additional keyword arguments.
   */
  abstract tokenize(inputs: Tensor1D): Tensor1D[];

  /**
   * Transform tokens back into strings.
   *
   * @param inputs Input tensor.
   * @param kwargs Additional keyword arguments.
   */
  detokenize(inputs: Tensor1D[]): Tensor1D {
    throw new NotImplementedError(
      `No implementation of 'detokenize()' was found for
      ${this.constructor.name}.`
    );
  }

  /**
   * Get the tokenizer vocabulary as a list of strings terms.
   */
  get vocabulary(): string[] {
    throw new NotImplementedError(
      `No implementation of 'vocabulary()' was found for
      ${this.constructor.name}.`
    );
  }

  /**
   * Returns the total size of the token id space.
   */
  get vocabularySize(): number {
    throw new NotImplementedError(
      `No implementation of 'vocabularySize()' was found for
      ${this.constructor.name}.`
    );
  }

  /**
   * Convert an integer id to a string token.
   */
  idToToken(id: number): string {
    throw new NotImplementedError(
      `No implementation of 'idToToken()' was found for
      ${this.constructor.name}.`
    );
  }

  /**
   * Convert an integer id to a string token.
   */
  tokenToId(token: string): number {
    throw new NotImplementedError(
      `No implementation of 'tokenToId()' was found for
      ${this.constructor.name}.`
    );
  }

  override call(inputs: Tensor1D|Tensor1D[], kwargs: TokenizerOptions={mode: 'tokenize'}): Tensor1D|Tensor1D[] {
    if (kwargs.mode === 'tokenize') {
      if (inputs instanceof Array) {
        throw new ValueError(`tokenize expects Tensor1D, not Tensor1D[].`);
      }
      return this.tokenize(inputs);
    }

    if (kwargs.mode === 'detokenize') {
      if (!(inputs instanceof Array)) {
        throw new ValueError(`detokenize expects Tensor1D[], not Tensor1D.`);
      }
      return this.detokenize(inputs);
    }

    throw new ValueError(`Input mode=${kwargs.mode} is not supported.`)
  }
}

export class WhiteSpaceTokenizer extends Tokenizer {
  /** @nocollapse */
  static readonly className = 'WhiteSpaceTokenizer';

  tokenize(inputs: Tensor1D): Tensor1D[] {
    const stringInputs = inputs.dataSync() as unknown as string[];
    return stringInputs.map(input => tensor1d(input.split(' ')));
  }

  override detokenize(inputs: Tensor1D[]): Tensor1D {
    const stringInputs = inputs.map(input => input.dataSync() as unknown as string[]);
    return tensor1d(stringInputs.map(str => str.join(' ')));
  }
}

serialization.registerClass(WhiteSpaceTokenizer);
