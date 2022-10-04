/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { describeMathCPUAndGPU, expectTensorsClose} from '../../utils/test_utils';
import { Tensor, tensor} from '@tensorflow/tfjs-core';
import { CategoryEncoding } from './category_encoding';

describeMathCPUAndGPU('Category Encoding', () => {

   it('Calculates correct output for Count outputMode rank 0', () => {
    const categoryData = tensor(0);
    const expectedOutput = tensor([1,0,0,0]);
    const numTokens = 4;
    const encodingLayer = new CategoryEncoding({numTokens,
                                outputMode: 'count'});
    const computedOutput = encodingLayer.
                          apply(categoryData) as Tensor;

    expectTensorsClose(computedOutput, expectedOutput);
  });

  it('Calculates correct output for Count outputMode rank 1 (weights)', () => {
    const categoryData = tensor([1, 2, 3, 3, 0]);
    const weightData = tensor([1, 2, 3, 1, 7]);
    const numTokens = 6;
    const expectedOutput = tensor([7, 1, 2, 4, 0, 0]);
    const encodingLayer = new CategoryEncoding({numTokens,
                                            outputMode: 'count'});

    const computedOutput = encodingLayer.apply(categoryData,
                      {countWeights: weightData}) as Tensor;

    expectTensorsClose(computedOutput, expectedOutput);
  });

  it('Calculates correct output for Count outputMode rank 2', () => {
    const categoryData   = tensor([[1, 2, 3, 1], [0, 3, 1, 0]]);
    const expectedOutput = tensor([[0, 2, 1, 1, 0, 0], [2, 1, 0, 1, 0, 0]]);
    const numTokens = 6;
    const encodingLayer = new CategoryEncoding({numTokens,
                                            outputMode: 'count'});
    const computedOutput = encodingLayer.apply(categoryData) as Tensor;
    expectTensorsClose(computedOutput, expectedOutput);
  });

  it('Calculates correct output for oneHot outputMode rank 0', () => {
    const categoryData = tensor(3);
    const expectedOutput = tensor([0, 0, 0, 1]);
    const numTokens = 4;
    const encodingLayer = new CategoryEncoding({numTokens,
                                          outputMode: 'oneHot'});
    const computedOutput = encodingLayer.apply(categoryData) as Tensor;
    expectTensorsClose(computedOutput, expectedOutput);
  });

  it('Calculates correct output and shape for oneHot outputMode rank 1', () => {
    const categoryData   = tensor([3, 2, 0, 1]);
    const expectedOutput = tensor([[0, 0, 0, 1],
                                   [0, 0, 1, 0],
                                   [1, 0, 0, 0],
                                   [0, 1, 0, 0]]);
    const numTokens = 4;
    const encodingLayer = new CategoryEncoding({numTokens,
                                          outputMode: 'oneHot'});
    const computedOutput = encodingLayer.apply(categoryData) as Tensor;
    expectTensorsClose(computedOutput, expectedOutput);
  });

  it('Calculates correct output and shape for oneHot outputMode rank 2', () => {
    const categoryData   = tensor([[3], [2], [0], [1]]);
    const expectedOutput = tensor([[0, 0, 0, 1],
                                   [0, 0, 1, 0],
                                   [1, 0, 0, 0],
                                   [0, 1, 0, 0]]);
    const numTokens = 4;
    const encodingLayer = new CategoryEncoding({numTokens,
                                          outputMode: 'oneHot'});
    const computedOutput = encodingLayer.apply(categoryData) as Tensor;
    expectTensorsClose(computedOutput, expectedOutput);
  });

  it('Calculates correct output for multiHot outputMode rank 0', () => {
    const categoryData = tensor(3);
    const expectedOutput = tensor([0, 0, 0, 1, 0, 0]);
    const numTokens = 6;
    const encodingLayer = new CategoryEncoding({numTokens,
                                outputMode: 'multiHot'});
    const computedOutput = encodingLayer.apply(categoryData) as Tensor;
    expectTensorsClose(computedOutput, expectedOutput);
  });

  it('Calculates correct output for multiHot outputMode rank 1', () => {
    const categoryData   = tensor([3, 2, 0, 1]);
    const expectedOutput = tensor([1, 1, 1, 1, 0, 0]);
    const numTokens = 6;
    const encodingLayer = new CategoryEncoding({numTokens,
                                        outputMode: 'multiHot'});
    const computedOutput = encodingLayer.apply(categoryData) as Tensor;
    expectTensorsClose(computedOutput, expectedOutput);
  });

  it('Calculates correct output for multiHot outputMode rank 2', () => {
    const categoryData   = tensor([[0, 1], [0, 0], [1, 2], [3, 1]]);
    const expectedOutput = tensor([[1, 1, 0, 0],
                                   [1, 0, 0, 0],
                                   [0, 1, 1, 0],
                                   [0, 1, 0, 1]]);
    const numTokens = 4;
    const encodingLayer = new CategoryEncoding({numTokens,
                                        outputMode: 'multiHot'});
    const computedOutput = encodingLayer.apply(categoryData) as Tensor;
    expectTensorsClose(computedOutput, expectedOutput);
  });

  it('Raises Value Error if input Tensor has Rank > 2', () =>{
    const categoryData = tensor([[[1], [2]], [[3], [4]]]);
    const numTokens = 6;
    const encodingLayer = new CategoryEncoding({numTokens,
                                        outputMode: 'multiHot'});
    expect(() => encodingLayer.apply(categoryData))
    .toThrowError(`When outputMode is not int, maximum output rank is 2`
    + ` Received outputMode ${'multiHot'} and input shape ${categoryData.shape}`
    + ` which would result in output rank ${categoryData.rank}.`);
  });

  it('Raises Value Error if max input value !<= numTokens', () => {
    const categoryData   = tensor([7, 2, 0, 1]);
    const numTokens = 3;
    const encodingLayer = new CategoryEncoding({numTokens,
                                        outputMode: 'multiHot'});
    expect(() => encodingLayer.apply(categoryData))
    .toThrowError('Input values must be between 0 < values <= numTokens'
      + ` with numTokens=${numTokens}`);
  });

  it('Raises Value Error if min input value < 0', () => {
    const categoryData   = tensor([7, 2, -1, 1]);
    const numTokens = 3;
    const encodingLayer = new CategoryEncoding({numTokens,
                                        outputMode: 'multiHot'});
    expect(() => encodingLayer.apply(categoryData))
    .toThrowError(`Input values must be between 0 < values <= numTokens`
      + ` with numTokens=${numTokens}`);
  });
});
