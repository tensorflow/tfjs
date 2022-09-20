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




 }
