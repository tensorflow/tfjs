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
import { Tensor, serialization } from '@tensorflow/tfjs-core';

import { Layer, LayerArgs } from '../../../engine/topology';
import { NotImplementedError } from '../../../errors';

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

  /**
   * Boolean. Whether to return a boolean padding mask of all locations that are
   * filled in with the `padValue`.
   * Defaults to false.
   */
  returnPaddingMask?: boolean;
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
  private startValue: number|string;
  private endValue: number|string;
  private padValue: number|string;
  private returnPaddingMask: boolean;

  constructor(args: StartEndPackerArgs) {
    super(args);

    this.sequenceLength = args.sequenceLength;
    this.startValue = args.startValue || null;
    this.endValue = args.endValue || null;
    this.padValue = args.padValue || null;
    this.returnPaddingMask = args.returnPaddingMask || false;
  }

  override call(inputs: Tensor|Tensor[]): Tensor|Tensor[] {
    throw new NotImplementedError(`Call method not implemented `);
  }

  override getConfig(): serialization.ConfigDict {
    const config = {
      sequence_length: this.sequenceLength,
      start_value: this.startValue,
      end_value: this.endValue,
      pad_value: this.padValue,
      return_padding_mask: this.returnPaddingMask,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(StartEndPacker);
