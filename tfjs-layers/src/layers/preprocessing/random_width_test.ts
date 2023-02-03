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

import { Tensor, reshape, range, image, Rank, zeros, randomUniform } from "@tensorflow/tfjs-core";
import { describeMathCPUAndGPU, expectTensorsClose } from "../../utils/test_utils";

import { RandomWidth, RandomWidthArgs } from "./random_width";

describeMathCPUAndGPU("RandomWidth Layer", () => {
  it("Returns correct, randomly scaled width of Rank 3 Tensor", () => {
    const rangeTensor = range(0, 16);
    const inputTensor = reshape(rangeTensor, [4,4,1]);
    const factor = 0.4;
    const interpolation = 'nearest';
    const seed = 42;
    const randomWidthLayer = new RandomWidth({factor, interpolation, seed});
    const layerOutputTensor = randomWidthLayer.apply(inputTensor) as Tensor;

    expect(layerOutputTensor.shape).toEqual([4,5,1]);
  });

  it("Returns correct, randomly scaled width of Rank 4 Tensor", () => {
    const rangeTensor = range(0, 16);
    const inputTensor = reshape(rangeTensor, [1,4,4,1]);
    const factor = 0.4;
    const interpolation = 'nearest';
    const seed = 42;
    const randomWidthLayer = new RandomWidth({factor, interpolation, seed});
    const layerOutputTensor = randomWidthLayer.apply(inputTensor) as Tensor;

    expect(layerOutputTensor.shape).toEqual([1,4,5,1]);
  });
  
  it('Returns a tensor of the correct dtype and compares class results to standard ops', () => {
    // perform same random width resizing operations, check tensors dtypes and content
    const inputTensor: Tensor<Rank.R3> =
    zeros([4, 4, 1]);
    const factor = 0.4;
    const interpolation = 'nearest';
    const seed = 42;
    const randomWidthLayer = new RandomWidth({factor, interpolation, seed});
    const layerOutputTensor = randomWidthLayer.apply(inputTensor) as Tensor;
    const widthFactor = randomUniform([1], 1 + -factor, 1 + factor, 'float32', seed)
    const adjustedWidth = Math.round(widthFactor.dataSync()[0] * 4);
    const size: [number, number] = [4, adjustedWidth]
    const expectedOutputTensor = image.resizeNearestNeighbor(inputTensor, size);
    expect(layerOutputTensor.dtype).toBe(inputTensor.dtype)
    expectTensorsClose(layerOutputTensor, expectedOutputTensor)
  });

  it("Check interpolation method 'unimplemented'", () => {
    const factor = 0.5;
    const interpolation = 'unimplemented';
    const seed = 42;
    const incorrectArgs = {factor, interpolation, seed }
    const expectedError = `Invalid interpolation parameter: ${
    interpolation} is not implemented`;
    expect(() => new RandomWidth(incorrectArgs as RandomWidthArgs))
      .toThrowError(expectedError)
  });

  it('Config holds correct name', () => {
    // layer name property set properly
    const factor = 0.4;
    const interpolation = 'nearest';
    const seed = 42;  
    const randomWidthLayer = new RandomWidth({factor, interpolation, seed, name: 'RandomWidth'});
    const config = randomWidthLayer.getConfig();
    expect(config.name).toEqual('RandomWidth');
  });
});
