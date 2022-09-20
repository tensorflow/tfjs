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
 import { serialization, Tensor, tidy } from '@tensorflow/tfjs-core';
 import { Shape } from '../../keras_format/common';
 import { getExactlyOneShape, getExactlyOneTensor } from '../../utils/types_utils';
 import { Kwargs } from '../../types';
 import * as K from '../../backend/tfjs_backend';
import { ValueError } from '../../errors';





 export declare interface CategoryEncodingArgs extends LayerArgs {
  numTokens: number;
  outputMode?: string;
  sparse?: boolean;

 }

 export abstract class CategoryEncoding extends Layer {
  /** @nocollapse */
  static className = 'CategoryEncoding';
  private readonly numTokens: number;
  private readonly outputMode: string;
  private readonly sparse: boolean;

  constructor(args: CategoryEncodingArgs) {
    super(args);
    this.numTokens = args.numTokens;

    if(args.outputMode) {
    this.outputMode = args.outputMode;
    } else {
      this.outputMode = "multiHot";
    }

    if(args.sparse) {
      this.sparse = args.sparse;
      } else {
        this.sparse = false;
      }
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      'numTokens': this.numTokens,
      'outputMode': this.outputMode,
      'sparse': this.sparse
    };

    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);

    if(inputShape == null) {
      return [this.numTokens]
    }

    if(this.outputMode == "oneHot" && inputShape[-1] !== 1) {
      inputShape.push(this.numTokens)
      return inputShape
    }

    inputShape[-1] = this.numTokens
    return inputShape
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor[]|Tensor {
    return tidy(() => {
      inputs = getExactlyOneTensor(inputs);
      if(inputs.dtype !== 'int32') {
          inputs = K.cast(inputs, 'int32');
      }
      if(kwargs && kwargs["countWeights"]) {

        if(this.outputMode !== "count") {
          throw new ValueError(`countWeights is not used when outputMode is not
          count. Received countWeights=${kwargs["countWeights"]}`)
        }

        const countWeights = getExactlyOneTensor(kwargs["countWeights"])
      }

      /// convert sparseToDense if necessary?
      const depth = this.numTokens



      return inputs;
    });
  }
}


