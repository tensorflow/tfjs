/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { LayerArgs, Layer } from '../../engine/topology';
import { serialization, Tensor, tidy, Tensor1D, Tensor2D} from '@tensorflow/tfjs-core';
import { greater, greaterEqual, max, min} from '@tensorflow/tfjs-core';
import { Shape } from '../../keras_format/common';
import { getExactlyOneShape, getExactlyOneTensor } from '../../utils/types_utils';
import { Kwargs } from '../../types';
import { ValueError } from '../../errors';
import * as K from '../../backend/tfjs_backend';
import * as utils from './preprocessing_utils';

export declare interface CategoryEncodingArgs extends LayerArgs {
  numTokens: number;
  outputMode?: OutputMode;
 }

export class CategoryEncoding extends Layer {
  /** @nocollapse */
  static className = 'CategoryEncoding';
  private readonly numTokens: number;
  private readonly outputMode: OutputMode;

  constructor(args: CategoryEncodingArgs) {
    super(args);
    this.numTokens = args.numTokens;

    if(args.outputMode) {
    this.outputMode = args.outputMode;
    } else {
      this.outputMode = utils.multiHot;
    }
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      'numTokens': this.numTokens,
      'outputMode': this.outputMode,
    };

    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);

    if(inputShape == null) {
      return [this.numTokens];
    }

    if(this.outputMode === utils.oneHot && inputShape[-1] !== 1) {
      inputShape.push(this.numTokens);
      return inputShape;
    }

    inputShape[-1] = this.numTokens;
    return inputShape;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor[]|Tensor {
    return tidy(() => {

      inputs = getExactlyOneTensor(inputs);
      if(inputs.dtype !== 'int32') {
        inputs = K.cast(inputs, 'int32');
    }

       let countWeights;

      if((typeof kwargs['countWeights']) !== 'undefined') {

        if(this.outputMode !== utils.count) {
          throw new ValueError(
            `countWeights is not used when outputMode !== count.
             Received countWeights=${kwargs['countWeights']}`);
        }
         const countWeightsRanked = getExactlyOneTensor(kwargs['countWeights']);

         if(countWeightsRanked.rank === 1) {
           countWeights = countWeightsRanked as Tensor1D;
         } if(countWeightsRanked.rank === 2) {
           countWeights = countWeightsRanked as Tensor2D;
          }
      }

      const depth = this.numTokens;
      const maxValue = max(inputs);
      const minValue = min(inputs);

      const greaterEqualMax = greater(depth, maxValue).bufferSync().get(0);
      const greaterMin = greaterEqual(minValue, 0).bufferSync().get(0);

      if(!(greaterEqualMax && greaterMin)) {
        throw new ValueError(
        `Input values must be between 0 < values <= numTokens`);
      }

    return utils.encodeCategoricalInputs(inputs,
      this.outputMode, depth, countWeights);
    });
  }
}

serialization.registerClass(CategoryEncoding);
