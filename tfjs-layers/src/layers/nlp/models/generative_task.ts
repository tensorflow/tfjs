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
 *  Base class for Generative Task models.
 */

/* Original source: keras_nlp/models/generative_task.py */
import { NamedTensorMap, Tensor } from '@tensorflow/tfjs-core';

import { NotImplementedError } from '../../../errors';
import { ModelCompileArgs } from '../../../engine/training';

import { Task } from './task';

export type GenerateFn =
  (inputs: NamedTensorMap, endTokenId?: number) => NamedTensorMap;

/**
 *  Base class for Generative Task models.
 */
export class GenerativeTask extends Task {
  /** @nocollapse */
  static override className = 'GenerativeTask';

  protected generateFunction: GenerateFn;

  override compile(args: ModelCompileArgs): void {
    throw new NotImplementedError();
  }

  /**
   * Run the generation on a single batch of input.
   */
  generateStep(
    inputs: NamedTensorMap,
    endTokenId: number
  ): NamedTensorMap {
    throw new NotImplementedError();
  }

  /**
   * Create or return the compiled generation function.
   */
  makeGenerateFunction(): GenerateFn {
    throw new NotImplementedError();
  }

  /**
   * Normalize user input to the generate function.
   *
   * This function converts all inputs to tensors, adds a batch dimension if
   * necessary, and returns a iterable "dataset like" object.
   */
  protected normalizeGenerateInputs(inputs: Tensor): [Tensor, boolean] {
    throw new NotImplementedError();
  }

  /**
   * Normalize user output from the generate function.
   *
   * This function converts all output to numpy (for integer output), or
   * python strings (for string output). If a batch dimension was added to
   * the input, it is removed from the output (so generate can be string in,
   * string out).
   */
  protected normalizeGenerateOutputs(
    outputs: Tensor,
    inputIsScalar: boolean
  ): Tensor {
    throw new NotImplementedError();
  }

  /**
   * Generate text given prompt `inputs`.
   *
   * This method generates text based on given `inputs`. The sampling method
   * used for generation can be set via the `compile()` method.
   *
   * `inputs` will be handled as a single batch.
   *
   * If a `preprocessor` is attached to the model, `inputs` will be
   * preprocessed inside the `generate()` function and should match the
   * structure expected by the `preprocessor` layer (usually raw strings).
   * If a `preprocessor` is not attached, inputs should match the structure
   * expected by the `backbone`. See the example usage above for a
   * demonstration of each.
   *
   * @param inputs tensor data. If a `preprocessor` is attached to the model,
   *  `inputs` should match the structure expected by the `preprocessor` layer.
   *  If a `preprocessor` is not attached, `inputs` should match the structure
   *  expected the the `backbone` model.
   * @param maxLength Integer. The max length of the generated sequence.
   *  Will default to the max configured `sequenceLength` of the
   *  `preprocessor`. If `preprocessor` is `null`, `inputs` should be
   *  should be padded to the desired maximum length and this argument
   *  will be ignored.
   */
  generate(inputs: Tensor, maxLength?: number) {
    throw new NotImplementedError();
  }
}
