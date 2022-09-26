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

import {image, randomNormal, Rank, Tensor} from '@tensorflow/tfjs-core';

// import {Shape} from '../../keras_format/common';
import {describeMathCPUAndGPU, expectTensorsClose} from '../../utils/test_utils';

import {Resizing} from './image_resizing';

describeMathCPUAndGPU('Resizing Layer', () => {
  it('Check if output shape matches specifications', () => {
    // resize and check output shape
    const maxHeight = 40;
    const height = Math.floor(Math.random() * maxHeight);
    const maxWidth = 60;
    const width = Math.floor(Math.random() * maxWidth);
    const numChannels = 3;
    const inputTensor = randomNormal([height * 2, width * 2, numChannels]);
    const expectedOutputShape = [height, width, numChannels];
    const resizingLayer = new Resizing({height, width});
    const layerOutputTensor = resizingLayer.apply(inputTensor) as Tensor;
    expect(expectedOutputShape).toEqual(layerOutputTensor.shape);
  });

  it('Returns correctly resized tensor', () => {
    // resize twice; once to make it smaller, once to return it to original size
    const maxHeight = 100;
    const height = Math.floor(Math.random() * maxHeight);
    const maxWidth = 100;
    const width = Math.floor(Math.random() * maxWidth);
    const numChannels = 3;
    const inputTensor = randomNormal([maxHeight, maxWidth, numChannels]);
    const resizingFirst = new Resizing({height, width});
    const layerFirstTensor = resizingFirst.apply(inputTensor) as Tensor;
    const resizingSecond = new Resizing({height: maxHeight, width: maxWidth});
    const layerOutputTensor = resizingSecond.apply(layerFirstTensor) as Tensor;
    expectTensorsClose(inputTensor, layerOutputTensor);
  });

  it('Returns the same tensor when given same shape as input', () => {
    // create a resizing layer with same shape as input
    const height = 64;
    const width = 32;
    const numChannels = 3;
    const inputTensor = randomNormal([height, width, numChannels]);
    const resizingLayer = new Resizing({height, width});
    const layerOutputTensor = resizingLayer.apply(inputTensor) as Tensor;
    expectTensorsClose(inputTensor, layerOutputTensor);
  });

  it('Returns a tensor of the correct dtype', () => {
    // do a same resizing operation, cheeck tensors dtypes and content
    const maxHeight = 40;
    const height = Math.floor(Math.random() * maxHeight);
    const maxWidth = 60;
    const width = Math.floor(Math.random() * maxWidth);
    const numChannels = 3;
    const inputTensor: Tensor<Rank.R3> =
        randomNormal([height * 2, width * 2, numChannels]);
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
    expect(() => new Resizing(incorrectArgs)).toThrowError(expectedError);
  });

  it('Config holds correct name', () => {
    const height = 40;
    const width = 60;
    const resizingLayer = new Resizing({height, width, name:'Resizing'});
    const config = resizingLayer.getConfig();
    expect(config.name).toEqual('Resizing');
  });
});
