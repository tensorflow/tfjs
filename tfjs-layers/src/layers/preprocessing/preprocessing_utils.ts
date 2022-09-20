import {  Tensor, denseBincount } from '@tensorflow/tfjs-core';
import { Identity } from '@tensorflow/tfjs-core';
import { getExactlyOneTensor } from '../../utils/types_utils';
import { ExpandDims } from '@tensorflow/tfjs-core';

export const int = "int"
export const oneHot = "oneHot"
export const multiHot = "multiHot"
export const count = "count"
export const tfIdf = "tfIdf"


export function encodeInputs(inputs: Tensor|Tensor[], outputMode: string,
                             depth: number, countWeights: Tensor|null,
                             idfWeights: Tensor|null): Tensor|Tensor[] {

  if(outputMode === oneHot || outputMode === multiHot) {
    return denseBincount()
  }

  return inputs
}
