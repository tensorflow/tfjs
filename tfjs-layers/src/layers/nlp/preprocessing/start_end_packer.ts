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
 *  Start End Packer implementation based on `tf.layers.Layer`.
 */

/* Original source: keras-nlp/start_end_packer.py */
import { Tensor, Tensor1D, Tensor2D, concat, serialization, stack, tensor, tidy } from '@tensorflow/tfjs-core';

import { Layer, LayerArgs } from '../../../engine/topology';
import { ValueError } from '../../../errors';

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
 * Adds start and end tokens to a sequence and pads to a fixed length.
 *
 *  This layer is useful when tokenizing inputs for tasks like translation,
 *  where each sequence should include a start and end marker. It should
 *  be called after tokenization. The layer will first trim inputs to fit, then
 *  add start/end tokens, and finally pad, if necessary, to `sequence_length`.
 *
 *  Input should be either a `tf.Tensor[]` or a dense `tf.Tensor`, and
 *  either rank-1 or rank-2.
 */
export class StartEndPacker extends Layer {
  /** @nocollapse */
  static readonly className = 'StartEndPacker';

  private sequenceLength: number;
  private startValue?: number|string;
  private endValue?: number|string;
  private padValue?: number|string;

  constructor(args: StartEndPackerArgs) {
    super(args);

    this.sequenceLength = args.sequenceLength;
    this.startValue = args.startValue;
    this.endValue = args.endValue;
    this.padValue = args.padValue;
  }

  override call(
    inputs: Tensor|Tensor[],
    kwargs: StartEndPackerOptions={addStartValue: true, addEndValue: true}
  ): Tensor|Tensor2D {
    return this.callAndReturnPaddingMask(inputs, kwargs)[0];
  }

  /**
   * Exactly like `call` except also returns a boolean padding mask of all
   * locations that are filled in with the `padValue`.
   */
  callAndReturnPaddingMask(
    inputs: Tensor|Tensor[],
    kwargs: StartEndPackerOptions={addStartValue: true, addEndValue: true}
  ): [Tensor1D|Tensor2D, Tensor1D|Tensor2D] {
    return tidy(() => {
      // Add a new axis at the beginning if needed.
      let x = inputs instanceof Tensor ? [inputs] : inputs;

      const inputIs1d = inputs instanceof Tensor && inputs.rank === 1;

      if (x.some(t => t.rank !== 1)) {
        throw new ValueError(
          'Input must either be a rank 1 Tensor or an array of rank 1 Tensors.'
        );
      }
      const sequenceLength = kwargs.sequenceLength ?? this.sequenceLength;

      // Concatenate start and end tokens.
      if (kwargs.addStartValue && this.startValue != null) {
        const startTokenIdTensor = tensor([this.startValue]);
        x = x.map(t => concat([startTokenIdTensor, t]));
      }
      if (kwargs.addEndValue && this.endValue != null) {
        const endTokenIdTensor = tensor([this.endValue]);
        // Trim to leave room for end token.
        x = x.map(t => {
          const sliced = t.slice(0, Math.min(t.shape[0], sequenceLength - 1));
          const padded = concat([sliced, endTokenIdTensor]);
          return padded;
        });
      }

      // tf.pad does not allow padding on Tensors with dtype='string'
      function ensureLength(
        input: Tensor, length: number, padValue?: string|number) {
        if (padValue === undefined) {
          padValue = input.dtype === 'string' ? '' : 0;
        }
        if (typeof padValue === 'number') {
          return input.pad([[0, length - input.size]], padValue);
        }

        const strInput = input.arraySync() as unknown as string[];

        if (strInput.length <= length) {
          const pads = Array(length - strInput.length).fill(padValue);
          return tensor(strInput.concat(pads));
        }

        return tensor(strInput.slice(0, strInput.length - length));
      }

      const paddedMask: Tensor[] = x.map(t => {
        // `onesLike` not used since it does not support string tensors.
        const ones = tensor(Array(t.shape[0]).fill(1));
        return ensureLength(ones, sequenceLength, 0).cast('bool');
      });
      const mask = inputIs1d ?
        paddedMask[0] as Tensor1D
        : stack(paddedMask) as Tensor2D;

      const paddedTensors: Tensor[] =
        x.map(t => ensureLength(t, sequenceLength, this.padValue));
      const outputs = inputIs1d ?
        paddedTensors[0] as Tensor1D
        : stack(paddedTensors) as Tensor2D;

      return [outputs, mask];
    });
  }

  override getConfig(): serialization.ConfigDict {
    const config = {
      sequenceLength: this.sequenceLength,
      startValue: this.startValue,
      endValue: this.endValue,
      padValue: this.padValue,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(StartEndPacker);
