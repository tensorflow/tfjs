import { Tensor, denseBincount, Tensor1D, Tensor2D, TensorLike, mul} from '@tensorflow/tfjs-core';
import { getExactlyOneTensor } from '../../utils/types_utils';
import { expandDims} from '@tensorflow/tfjs-core';
import { ValueError } from '../../errors';
import * as K from '../../backend/tfjs_backend';

export const int = 'int';
export const oneHot = 'oneHot';
export const multiHot = 'multiHot';
export const count = 'count';
export const tfIdf = 'tfIdf';

export function encodeCategoricalInputs(inputs: Tensor|Tensor[],
                                        outputMode: string,
                                        depth: number,
                                        weights?: Tensor1D|Tensor2D|TensorLike):
                                        Tensor|Tensor[] {

  inputs = getExactlyOneTensor(inputs);

  if(inputs.dtype !== 'int32') {
        inputs = K.cast(inputs, 'int32');
    }

  if(outputMode === int) {
    return inputs;
  }

  const originalShape = inputs.shape;

  if(inputs.rank === 0) {
    inputs = expandDims(inputs, -1);
  }

  if(outputMode === oneHot) {
    if(inputs.shape[inputs.shape.length - 1] !== 1) {
      inputs = expandDims(inputs, -1);
    }
  }

  if(inputs.rank > 2) {
    throw new ValueError(`When outputMode is not 'int', maximum output rank is 2
    Received outputMode ${outputMode} and input shape ${originalShape}
    which would result in output rank ${inputs.rank}.`);
  }

  const binaryOutput = [multiHot, oneHot].includes(outputMode);

  let denseBincountInput;

  if(inputs.rank === 1) {
    denseBincountInput = inputs as Tensor1D;
  }

  if(inputs.rank === 2) {
    denseBincountInput = inputs as Tensor2D;
  }

  let binCounts;

  if((typeof weights) !== 'undefined' && outputMode === count) {

    binCounts = denseBincount(denseBincountInput, weights, depth, binaryOutput);

   } else {

    binCounts = denseBincount(denseBincountInput, [], depth, binaryOutput);

   }

  if(outputMode !== tfIdf) {
    return binCounts;
  }

  if(weights === null || weights === undefined) {
    throw new ValueError(
      `When outputMode is 'tfIdf', idfWeights must be provided.`
      );
  } else {

    return mul(binCounts, weights);

  }
}
