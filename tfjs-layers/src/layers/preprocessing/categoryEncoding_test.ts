import { describeMathCPUAndGPU, expectTensorsClose} from '../../utils/test_utils';
import {Tensor, tensor} from '@tensorflow/tfjs-core';

import { CategoryEncoding } from './categoryEncoding';
import * as utils from '../../utils/preprocessing_utils'

describeMathCPUAndGPU('Category Encoding Layer', () => {

  it("Calculates correct output for Count outputMode", () => {
    console.log("RAN THIS TEST")
    const categoryData = tensor([1, 2, 3, 3, 0])
    const weightData = tensor([1, 2, 3, 1, 7])
    const expectedOutput = tensor([7, 1, 2, 4, 0, 0])
    const encodingLayer = new CategoryEncoding({numTokens: 6, outputMode: utils.count})
    const computedOutput = encodingLayer.apply(categoryData, {countWeights: weightData}) as Tensor
    console.log(`!!!!!!!!!!! COMPUTED OUTPUT OF TENSOR COUNT ${computedOutput}`)
    expectTensorsClose(expectedOutput, computedOutput)
  })


  it("Calculates correct output for oneHot outputMode", () => {
    console.log("RAN THIS TEST ONEHOT")
    const categoryData   = tensor([3, 2, 0, 1])
    const expectedOutput = tensor([[0, 0, 0, 1],
                                   [0, 0, 1, 0],
                                   [1, 0, 0, 0],
                                   [0, 1, 0, 0]])
    const encodingLayer = new CategoryEncoding({numTokens: 4, outputMode: utils.oneHot})
    const computedOutput = encodingLayer.apply(categoryData) as Tensor
    console.log(`!!!!!!!!!!! COMPUTED OUTPUT OF TENSOR ONEHOT ${computedOutput}`)
    expectTensorsClose(expectedOutput, computedOutput)
  })


  it("Calculates correct output for multiHot outputMode", () => {
    console.log("RAN THIS TEST MULTIHOT")
    const categoryData   = tensor([[0, 1], [0, 0], [1, 2], [3, 1]])
    const expectedOutput = tensor([[1, 1, 0, 0],
                                   [1, 0, 0, 0],
                                   [0, 1, 1, 0],
                                   [0, 1, 0, 1]])
    const encodingLayer = new CategoryEncoding({numTokens: 4, outputMode: utils.multiHot})
    const computedOutput = encodingLayer.apply(categoryData) as Tensor
    console.log(`!!!!!!!!!!! COMPUTED OUTPUT OF TENSOR multiHot ${computedOutput}`)
    expectTensorsClose(expectedOutput, computedOutput)
  })
})
