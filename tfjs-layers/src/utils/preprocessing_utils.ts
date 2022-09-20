import { Tensor, denseBincount, Tensor1D, Tensor2D, TensorLike, mul} from '@tensorflow/tfjs-core';
import { getExactlyOneTensor } from './types_utils';
import { expandDims } from '@tensorflow/tfjs-core';
import { ValueError } from '../errors';

export const int = "int"
export const oneHot = "oneHot"
export const multiHot = "multiHot"
export const count = "count"
export const tfIdf = "tfIdf"


export function encodeCategoricalInputs(inputs: Tensor|Tensor[], outputMode: string,
                             depth: number, countWeights: Tensor1D|Tensor2D|TensorLike,
                             idfWeights: Tensor1D|Tensor2D|null): Tensor|Tensor[] {

  inputs = getExactlyOneTensor(inputs)

  if(outputMode === int) {
    return inputs
  }

  const originalShape = inputs.shape

  if(inputs.rank === 0) {
    inputs = expandDims(inputs, -1)
  }
  if(outputMode === oneHot) {
    if(inputs.shape[-1] !== 1) {
      inputs = expandDims(inputs, -1)
    }
  }
  if(inputs.rank > 2) {
    throw new ValueError(`When output_mode is not 'int', maximum supported output rank is 2.
    Received output_mode ${outputMode} and input shape ${originalShape}
    which would result in output rank ${inputs.rank}.`)
  }
  const binaryOutput = [multiHot, oneHot].includes(outputMode)
  let denseBincountInput

  if(inputs.rank === 1) {
    denseBincountInput = inputs as Tensor1D
  }
  if(inputs.rank === 2) {
    denseBincountInput = inputs as Tensor2D
  }
  const binCounts = denseBincount(denseBincountInput, countWeights, depth, binaryOutput)

  if(outputMode !== tfIdf) {
    return binCounts
  }

  if(idfWeights === null) {
    throw new ValueError(`When outputMode is 'tfIdf', idfWeights must be provided.
    Received: outputMode=${outputMode} and idfWeights=${idfWeights}`)
  } else {

    return mul(binCounts, idfWeights)
  }



}
