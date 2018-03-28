/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit tests for convolutional_depthwise.ts.
 */

// tslint:disable:max-line-length
import {Tensor, tensor4d} from '@tensorflow/tfjs-core';

import * as K from '../backend/tfjs_backend';
import {DataFormat, PaddingMode} from '../common';
import {InitializerIdentifier} from '../initializers';
import {DType} from '../types';
import {SymbolicTensor} from '../types';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {DepthwiseConv2D} from './convolutional_depthwise';

// tslint:enable:max-line-length

describeMathCPU('DepthwiseConv2D-Symbolic', () => {
  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];
  const kernelSizes: [number, [number, number]] = [2, [2, 2]];
  const depthMultipliers = [1, 3];
  const paddingModes: PaddingMode[] = ['valid', 'same'];

  for (const dataFormat of dataFormats) {
    for (const kernelSize of kernelSizes) {
      for (const depthMultiplier of depthMultipliers) {
        for (const padding of paddingModes) {
          const testTitle = `dataFormat=${dataFormat}, ` +
              `kernelSize=${JSON.stringify(kernelSize)}, ` +
              `depthMultiplier=${depthMultiplier}, ` +
              `paddingMode=${padding}`;
          it(testTitle, () => {
            const depthwiseConvLayer = new DepthwiseConv2D(
                {dataFormat, kernelSize, depthMultiplier, padding});
            const inputShape = dataFormat === 'channelsFirst' ? [1, 8, 10, 10] :
                                                                [1, 10, 10, 8];
            const symbolicInput =
                new SymbolicTensor(DType.float32, inputShape, null, [], null);
            const symbolicOutput =
                depthwiseConvLayer.apply(symbolicInput) as SymbolicTensor;

            const outputImageSize = padding === 'valid' ? 9 : 10;
            let expectedShape: [number, number, number, number];
            if (dataFormat === 'channelsFirst') {
              expectedShape =
                  [1, 8 * depthMultiplier, outputImageSize, outputImageSize];
            } else {
              expectedShape =
                  [1, outputImageSize, outputImageSize, 8 * depthMultiplier];
            }
            expect(symbolicOutput.shape).toEqual(expectedShape);
          });
        }
      }
    }
  }

  it('Non-4D Array Input leads to exception', () => {
    const depthwiseConvLayer = new DepthwiseConv2D({kernelSize: 2});
    const symbolicInput =
        new SymbolicTensor(DType.float32, [1, 10, 10], null, [], null);
    expect(() => depthwiseConvLayer.apply(symbolicInput))
        .toThrowError(
            /Inputs to DepthwiseConv2D should have rank 4\. Received .*/);
  });
});

describeMathCPUAndGPU('DepthwiseConv2D-Tensor:', () => {
  const x4by4Data = [[[
    [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
    [-20, -40, -60, -80]
  ]]];

  const depthMultipliers = [1, 2];
  const useBiases = [false];
  const biasInitializers: InitializerIdentifier[] = ['zeros', 'ones'];

  for (const depthMultiplier of depthMultipliers) {
    for (const useBias of useBiases) {
      for (const biasInitializer of biasInitializers) {
        const testTitle =
            `channelsFirst, depthMultiplier=${depthMultiplier}, ` +
            `useBias=${useBias}, biasInitializer=${biasInitializer}, ` +
            `activation=relu`;
        it(testTitle, () => {
          const x = tensor4d(x4by4Data, [1, 1, 4, 4]);
          const conv2dLayer = new DepthwiseConv2D({
            kernelSize: [2, 2],
            depthMultiplier,
            strides: [2, 2],
            dataFormat: 'channelsFirst',
            useBias,
            depthwiseInitializer: 'ones',
            biasInitializer,
            activation: 'relu'
          });
          const y = conv2dLayer.apply(x) as Tensor;

          let yExpectedShape: [number, number, number, number];
          let yExpectedData: number[];
          if (depthMultiplier === 1) {
            yExpectedShape = [1, 1, 2, 2];
            yExpectedData = [100, 260, -100, -260];
          } else if (depthMultiplier === 2) {
            yExpectedShape = [1, 2, 2, 2];
            yExpectedData = [100, 260, -100, -260, 100, 260, -100, -260];
          }
          if (useBias && biasInitializer === 'ones') {
            yExpectedData = yExpectedData.map(element => element + 1);
          }
          // relu.
          yExpectedData =
              yExpectedData.map(element => element >= 0 ? element : 0);
          const yExpected = tensor4d(yExpectedData, yExpectedShape);
          expectTensorsClose(y, yExpected);
        });
      }
    }
  }

  it('channelsLast', () => {
    // Convert input to channelsLast.
    const x = K.transpose(tensor4d(x4by4Data, [1, 1, 4, 4]), [0, 2, 3, 1]);
    const conv2dLayer = new DepthwiseConv2D({
      depthMultiplier: 2,
      kernelSize: [2, 2],
      strides: [2, 2],
      dataFormat: 'channelsLast',
      useBias: false,
      depthwiseInitializer: 'ones',
      activation: 'linear'
    });
    const y = conv2dLayer.apply(x) as Tensor;
    const yExpected =
        tensor4d([100, 100, 260, 260, -100, -100, -260, -260], [1, 2, 2, 2]);
    expectTensorsClose(y, yExpected);
  });
});
