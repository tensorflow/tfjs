/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit Tests for image resizing layer.
 */

import {image, Rank, Tensor, tensor, zeros, range, reshape} from '@tensorflow/tfjs-core';

// import {Shape} from '../../keras_format/common';
import {describeMathCPUAndGPU, expectTensorsClose} from '../../utils/test_utils';

import {Resizing, ResizingArgs} from './image_resizing';

describeMathCPUAndGPU('Resizing Layer', () => {
  it('Check if output shape matches specifications', () => {
    // resize and check output shape
    const maxHeight = 40;
    const height = Math.floor(Math.random() * maxHeight);
    const maxWidth = 60;
    const width = Math.floor(Math.random() * maxWidth);
    const numChannels = 3;
    const inputTensor = zeros([height * 2, width * 2, numChannels]);
    const expectedOutputShape = [height, width, numChannels];
    const resizingLayer = new Resizing({height, width});
    const layerOutputTensor = resizingLayer.apply(inputTensor) as Tensor;
    expect(layerOutputTensor.shape).toEqual(expectedOutputShape);
  });

  it('Returns correctly downscaled tensor', () => {
    // resize and check output content (not batched)
    const rangeTensor = range(0, 16);
    const inputTensor = reshape(rangeTensor, [4,4,1]);
    const height = 2;
    const width = 2;
    const interpolation = 'nearest';
    const resizingLayer = new Resizing({height, width, interpolation});
    const layerOutputTensor = resizingLayer.apply(inputTensor) as Tensor;
    const expectedArr = [[0, 3], [12, 15]];
    const expectedOutput = tensor(expectedArr, [2,2,1]);
    expectTensorsClose(layerOutputTensor, expectedOutput);
  });

  it('Returns correctly downscaled tensor', () => {
    // resize and check output content (batched)
    const rangeTensor = range(0, 36);
    const inputTensor = reshape(rangeTensor, [1,6,6,1]);
    const height = 3;
    const width = 3;
    const interpolation = 'nearest';
    const resizingLayer = new Resizing({height, width, interpolation});
    const layerOutputTensor = resizingLayer.apply(inputTensor) as Tensor;
    const expectedArr = [[0,3,5], [18,21,23], [30,33,35]];
    const expectedOutput = tensor([expectedArr], [1,3,3,1]);
    expectTensorsClose(layerOutputTensor, expectedOutput);
  });

  it('Returns correctly upscaled tensor', () => {
    const rangeTensor = range(0, 4);
    const inputTensor = reshape(rangeTensor, [1, 2, 2, 1]);
    const height = 4;
    const width = 4;
    const interpolation = 'nearest';
    const resizingLayer = new Resizing({height, width, interpolation});
    const layerOutputTensor = resizingLayer.apply(inputTensor) as Tensor;
    const expectedArr = [[0,0,1,1], [0,0,1,1], [2,2,3,3], [2,2,3,3]];
    const expectedOutput = tensor([expectedArr], [1,4,4,1]);
    expectTensorsClose(layerOutputTensor, expectedOutput);
  });

  it('Returns the same tensor when given same shape as input', () => {
    // create a resizing layer with same shape as input
    const height = 64;
    const width = 32;
    const numChannels = 1;
    const rangeTensor = range(0, height * width);
    const inputTensor = reshape(rangeTensor, [height, width, numChannels]);
    const resizingLayer = new Resizing({height, width});
    const layerOutputTensor = resizingLayer.apply(inputTensor) as Tensor;
    expectTensorsClose(layerOutputTensor, inputTensor);
  });

  it('Returns a tensor of the correct dtype', () => {
    // do a same resizing operation, cheeck tensors dtypes and content
    const height = 40;
    const width = 60;
    const numChannels = 3;
    const inputTensor: Tensor<Rank.R3> =
        zeros([height, width, numChannels]);
    const size: [number, number] = [height, width];
    const expectedOutputTensor = image.resizeBilinear(inputTensor, size);
    const resizingLayer = new Resizing({height, width});
    const layerOutputTensor = resizingLayer.apply(inputTensor) as Tensor;
    expect(layerOutputTensor.dtype).toBe(inputTensor.dtype);
    expectTensorsClose(layerOutputTensor, expectedOutputTensor);
  });

  it('Throws an error given incorrect parameters', () => {
    // pass incorrect interpolation method string to layer init
    const height = 16;
    const width = 16;
    const interpolation = 'unimplemented';
    const incorrectArgs = {height, width, interpolation};
    const expectedError =
        `Invalid interpolation parameter: ${interpolation} is not implemented`;
    expect(() => new Resizing(incorrectArgs as ResizingArgs))
        .toThrowError(expectedError);
  });

  it('Config holds correct name', () => {
    // layer name property set properly
    const height = 40;
    const width = 60;
    const resizingLayer = new Resizing({height, width, name:'Resizing'});
    const config = resizingLayer.getConfig();
    expect(config.name).toEqual('Resizing');
  });
});
