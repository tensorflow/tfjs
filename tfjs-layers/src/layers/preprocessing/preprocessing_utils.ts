/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { Tensor, denseBincount, Tensor1D, Tensor2D, TensorLike, mul} from '@tensorflow/tfjs-core';
import { getExactlyOneTensor } from '../../utils/types_utils';
import { expandDims} from '@tensorflow/tfjs-core';
import { ValueError } from '../../errors';
import * as K from '../../backend/tfjs_backend';

export type OutputMode = 'int' | 'oneHot' | 'multiHot' | 'count' | 'tfIdf';

export function encodeCategoricalInputs(inputs: Tensor|Tensor[],
                                        outputMode: OutputMode,
                                        depth: number,
                                        weights?: Tensor1D|Tensor2D|TensorLike):
                                        Tensor|Tensor[] {

  let input = getExactlyOneTensor(inputs);

  if(input.dtype !== 'int32') {
    input = K.cast(input, 'int32');
    }

  if(outputMode === 'int') {
    return input;
  }

  const originalShape = input.shape;

  if(input.rank === 0) {
    input = expandDims(input, -1);
  }

  if(outputMode === 'oneHot') {
    if(input.shape[input.shape.length - 1] !== 1) {
      input = expandDims(input, -1);
    }
  }

  if(input.rank > 2) {
    throw new ValueError(`When outputMode is not int, maximum output rank is 2`
    + ` Received outputMode ${outputMode} and input shape ${originalShape}`
    + ` which would result in output rank ${input.rank}.`);
  }

  const binaryOutput = ['multiHot', 'oneHot'].includes(outputMode);

  const denseBincountInput = input as Tensor1D | Tensor2D;

  let binCounts: Tensor1D | Tensor2D;

  if ((typeof weights) !== 'undefined' && outputMode === 'count') {
    binCounts = denseBincount(denseBincountInput, weights, depth, binaryOutput);
   } else {
    binCounts = denseBincount(denseBincountInput, [], depth, binaryOutput);
   }

  if(outputMode !== 'tfIdf') {
    return binCounts;
  }

  if (weights) {
    return mul(binCounts, weights);
  } else {
      throw new ValueError(
        `When outputMode is 'tfIdf', weights must be provided.`
      );
  }
}
