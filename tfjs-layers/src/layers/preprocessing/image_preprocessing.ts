/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {LayerArgs, Layer} from '../../engine/topology';
import { serialization, Tensor, mul, add, tidy } from '@tensorflow/tfjs-core';
import { getExactlyOneTensor } from '../../utils/types_utils';
import * as K from '../../backend/tfjs_backend';
import { Kwargs } from '../../types';

export declare interface RescalingArgs extends LayerArgs {
  scale: number;
  offset?: number;
}

/**
 * Preprocessing Rescaling Layer
 *
 * This rescales images by a scaling and offset factor
 */
export class Rescaling extends Layer {
  /** @nocollapse */
  static className = 'Rescaling';
  private readonly scale: number;
  private readonly offset: number;
  constructor(args: RescalingArgs) {
    super(args);

    this.scale = args.scale;

    if(args.offset) {
    this.offset = args.offset;
    } else {
      this.offset = 0;
    }
  }

  override getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      'scale': this.scale,
      'offset': this.offset
    };

    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  override call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor[]|Tensor {
    return tidy(() => {
      inputs = getExactlyOneTensor(inputs);
      if(inputs.dtype !== 'float32') {
          inputs = K.cast(inputs, 'float32');
      }
      return add(mul(inputs, this.scale), this.offset);
    });
  }
}

serialization.registerClass(Rescaling);
