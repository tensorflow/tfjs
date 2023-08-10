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
import { Tensor, tensor } from '@tensorflow/tfjs-core';

import { NotImplementedError } from '../../../errors';
import { ModelCompileArgs } from '../../../engine/training';

import { Task } from './task';
import { GPT2CausalLMPreprocessor } from './gpt2/gpt2_causal_lm_preprocessor';
import { GPT2Tokenizer } from './gpt2/gpt2_tokenizer';

export type GPT2TensorMap = {
  [name: string]: Tensor;
};

export type GenerateFn =
  (inputs: GPT2TensorMap, endTokenId?: number) => GPT2TensorMap;

/**
 *  Base class for Generative Task models.
 */
export class GenerativeTask extends Task {
  protected generateFunction: GenerateFn;

  override compile(args: ModelCompileArgs): void {
    throw new NotImplementedError();
  }

  /**
   * Run the generation on a single batch of input.
   */
  generateStep(
    inputs: GPT2TensorMap,
    endTokenId: number
  ): GPT2TensorMap {
    throw new NotImplementedError();
  }

  /**
   * Create or return the compiled generation function.
   */
  makeGenerateFunction(): GenerateFn {
    if (this.generateFunction == null) {
      this.generateFunction = this.generateStep;
    }
    return this.generateFunction;
  }

  /**
   * Normalize user input to the generate function.
   *
   * This function converts all inputs to tensors, adds a batch dimension if
   * necessary, and returns a iterable "dataset like" object.
   */
  protected normalizeGenerateInputs(
    inputs: Tensor|GPT2TensorMap
  ): [Tensor, boolean] {
    let inputIsScalar = false;

    function normalize(x: string|string[]|Tensor): [Tensor, boolean] {
      let xIsScalar = false;
      if (typeof x === 'string' || Array.isArray(x)) {
        x = tensor(x);
      }
      if (x instanceof Tensor && x.rank === 0) {
        xIsScalar = true;
        x = x.reshape([1, ...x.shape]);
      }
      return [x, xIsScalar];
    }

    if (!(inputs instanceof Tensor)) {
      for (const key in inputs) {
        [inputs[key], inputIsScalar] = normalize(inputs[key]);
      }
    } else {
      [inputs, inputIsScalar] = normalize(inputs);
    }
    return [inputs as Tensor, inputIsScalar]
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
    outputs: Tensor|GPT2TensorMap,
    inputIsScalar: boolean
  ): Tensor {
    function normalize(x: Tensor): Tensor {
      return inputIsScalar ? x.squeeze([0]) : x;
    }
    if (!(outputs instanceof Tensor)) {
      const normalized: GPT2TensorMap = {};
      for (const key in outputs) {
        normalized[key] = normalize(outputs[key]);
      }
      return normalized.tokenIds;
    }
    return normalize(outputs);
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
  generate(inputs: Tensor, maxLength?: number): Tensor {
    // Setup our three main passes.
    // 1. Optionally preprocessing strings to dense integer tensors.
    // 2. Generate new tokens via a compiled function on dense tensors.
    // 3. Optionally postprocess dense integer tensors back to string.
    const generateFunction = this.makeGenerateFunction();
    let endTokenId: number;

    const preprocessor = this.preprocessor;
    if (preprocessor != null) {
      // TODO(pforderique): Add default `get endTokenId()` to `Tokenizer`.
      endTokenId = (this.preprocessor.tokenizer as GPT2Tokenizer).endTokenId;
    }

    function preprocess(x: Tensor) {
      // TODO(pforderique): Generalize for other models' preprocessors.
      return (preprocessor as GPT2CausalLMPreprocessor).generatePreprocess(
        x, maxLength
      );
    }

    function generate(x: Tensor) {
      return generateFunction({tokenIds: x, paddingMask: null}, endTokenId);
    }

    function postprocess(x: Tensor) {
      // TODO(pforderique): Generalize for other models' preprocessors.
      return (preprocessor as GPT2CausalLMPreprocessor).generatePostprocess(x);
    }

    // Normalize inputs, apply our three passes, and normalize outputs.
    let inputIsScalar: boolean;
    [inputs, inputIsScalar] = this.normalizeGenerateInputs(inputs);

    if (this.preprocessor != null) {
      inputs = preprocess(inputs).tokenIds;
    }

    let outputs = generate(inputs).tokenIds;

    if (this.preprocessor != null) {
      outputs = postprocess(outputs).tokenIds;
    }

    return this.normalizeGenerateOutputs(outputs, inputIsScalar);
  }
}
