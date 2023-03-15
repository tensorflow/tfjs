/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit Tests for random width layer.
 */

import { Tensor, range, image, Rank, zeros, randomUniform, tensor } from '@tensorflow/tfjs-core';
import { describeMathCPUAndGPU, expectTensorsClose } from '../../utils/test_utils';

import { RandomWidth, RandomWidthArgs } from './random_width';

describeMathCPUAndGPU('RandomWidth Layer', () => {
  it('Returns correct, randomly scaled width of Rank 3 Tensor', () => {
    const inputTensor = range(0, 16).reshape([4, 4, 1]);
    const factor = 0.4;
    const interpolation = 'nearest';
    const seed = 42;
    const randomWidthLayer = new RandomWidth({factor, interpolation, seed});
    const layerOutputTensor = randomWidthLayer.apply(inputTensor) as Tensor;

    expect(layerOutputTensor.shape).toEqual([4,5,1]);
  });

  it('Returns correct, randomly scaled width of Rank 4 Tensor', () => {
    const inputTensor = range(0, 16).reshape([1,4,4,1]);
    const factor = 0.4;
    const interpolation = 'nearest';
    const seed = 42;
    const randomWidthLayer = new RandomWidth({factor, interpolation, seed});
    const layerOutputTensor = randomWidthLayer.apply(inputTensor) as Tensor;

    expect(layerOutputTensor.shape).toEqual([1,4,5,1]);
  });

  it('Returns a tensor of the correct dtype and compares class to standard ops',
    () => {
    // perform random width resizing ops, check tensors dtypes and content
    const inputTensor: Tensor<Rank.R3> =
    zeros([4, 4, 1]);
    const factor = 0.4;
    const interpolation = 'nearest';
    const seed = 42;
    const randomWidthLayer = new RandomWidth({factor, interpolation, seed});
    const layerOutputTensor = randomWidthLayer.apply(inputTensor) as Tensor;
    const lower = 1 + -factor;
    const upper = 1 + factor;
    const widthFactor = randomUniform([1], lower, upper, 'float32', seed);
    const adjustedWidth = Math.round(widthFactor.dataSync()[0] * 4);
    const size: [number, number] = [4, adjustedWidth];
    const expectedOutputTensor = image.resizeNearestNeighbor(inputTensor, size);
    expect(layerOutputTensor.dtype).toBe(inputTensor.dtype);
    expectTensorsClose(layerOutputTensor, expectedOutputTensor);
  });

  it('Returns correctly scaled tensor; not batched', () => {
    // randomly resize width and check output content
    const inputTensor = range(0, 16).reshape([4, 4, 1]);
    const factor = 0.4;
    const interpolation = 'nearest';
    const seed = 42;
    const randomWidthLayer = new RandomWidth({factor, interpolation, seed});
    const layerOutputTensor = randomWidthLayer.apply(inputTensor) as Tensor;
    const expectedArr = [
      [0, 0, 1, 2, 3],
      [4, 4, 5, 6, 7],
      [8, 8, 9, 10, 11],
      [12, 12, 13, 14, 15]
    ];
    const expectedOutput = tensor(expectedArr, [4, 5, 1]);
    expectTensorsClose(layerOutputTensor, expectedOutput);
  });

  it('Check unimplemented interpolation method', () => {
    const factor = 0.5;
    const interpolation = 'unimplemented';
    const seed = 42;
    const incorrectArgs = {factor, interpolation, seed };
    const expectedError = `Invalid interpolation parameter: ${
    interpolation} is not implemented`;
    expect(() => new RandomWidth(incorrectArgs as RandomWidthArgs))
      .toThrowError(expectedError);
  });

  it('Validate thrown error when factor is less than -1', () => {
    // test for factor array, first item in array must be >= -1
    const factor = [-5, 0.3];
    const interpolation = 'nearest';
    const seed = 42;
    const incorrectArgs = { factor, interpolation, seed };
    const expectedError =  `factor must have values larger than -1. Got: ${factor}`;
    expect(
      () => new RandomWidth(incorrectArgs as RandomWidthArgs)
    ).toThrowError(expectedError);
  });

  it('Validate thrown error when factor contains more than 2 elements', () => {
    // test for factor array, first item in array must be >= -1
    const factor = [-5, 0.3, 7];
    const interpolation = 'nearest';
    const seed = 42;
    const incorrectArgs = { factor, interpolation, seed };
    const expectedError =
    `Invalid factor: ${factor}. Must be positive number or tuple of 2 numbers`;

    expect(
      () => new RandomWidth(incorrectArgs as RandomWidthArgs)
    ).toThrowError(expectedError);
  });

  it('Throws an error if factor is an array, upper width < lower width', () => {
    // test for upper width < lower width
    const factor = [0.5, 0.3];
    const interpolation = 'nearest';
    const seed = 42;
    const incorrectArgs = { factor, interpolation, seed };
    const expectedError = `factor cannot have upper bound less than lower bound.
        Got upper bound: ${factor[1]}.
        Got lower bound: ${factor[0]}
      `;
    expect(
      () => new RandomWidth(incorrectArgs as RandomWidthArgs)
    ).toThrowError(expectedError);
  });

  it('Throws an error if factor is a negative number', () => {
    // test factor, if factor is a number, it must be positive
    const factor = -0.5;
    const interpolation = 'nearest';
    const seed = 42;
    const incorrectArgs = { factor, interpolation, seed };
    const expectedError =
    `Invalid factor: ${factor}. Must be positive number or tuple of 2 numbers`;

    expect(
      () => new RandomWidth(incorrectArgs as RandomWidthArgs)
    ).toThrowError(expectedError);
  });

  it('Config holds correct name', () => {
    // layer name property set properly
    const factor = 0.4;
    const interpolation = 'nearest';
    const seed = 42;
    const randomWidthLayer = new RandomWidth({
      factor, interpolation, seed, name: 'RandomWidth'
    });
    const config = randomWidthLayer.getConfig();
    expect(config.name).toEqual('RandomWidth');
  });
});
