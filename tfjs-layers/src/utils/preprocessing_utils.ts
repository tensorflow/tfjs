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
  console.log(`INPUTS HERE ${inputs}`)
  console.log(`INPUTS SHAPE ${inputs.shape}`)
  console.log(`INPUT RANK HERE ${inputs.rank}`)
  if(outputMode === int) {
    return inputs
  }

  const originalShape = inputs.shape
  console.log(`INPUT SHAPE HERE ${inputs.shape}`)

  if(inputs.rank === 0) {
    inputs = expandDims(inputs, -1)
    console.log(`RANK 0 PRINT INPUT ${inputs}`)

  }
  if(outputMode === oneHot) {
    if(inputs.shape[-1] !== 1) {
      console.log(`inputs.shape[-1] = ${inputs.shape[-1]}`)
      console.log(`typeof inputs.shape[-1] = ${typeof inputs.shape[-1]}`)
      inputs = expandDims(inputs, -1)
      console.log(`0utputMode === oneHot && inputs.shape[-1] !== 1 PRINT INPUT ${inputs}`)
    }
  }
  if(inputs.rank > 2) {
    console.log(`INPUT RANK TOO BIG - INPUT ${inputs}`)
    console.log(`INPUT RANK TOO BIG - RANK ${inputs.rank}`)

    throw new ValueError(`When output_mode is not 'int', maximum supported output rank is 2.
    Received outputMode ${outputMode} and input shape ${originalShape}
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
