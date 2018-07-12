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
import * as tfc from '@tensorflow/tfjs-core';
import {Tensor, tensor4d, Tensor4D} from '@tensorflow/tfjs-core';

import {DataFormat, PaddingMode} from '../common';
import * as tfl from '../index';
import {InitializerIdentifier} from '../initializers';
import {convertPythonicToTs, convertTsToPythonic} from '../utils/serialization_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {depthwiseConv2d} from './convolutional_depthwise';

// tslint:enable:max-line-length
describeMathCPUAndGPU('depthwiseConv2d', () => {
  const x4by4Data = [[[
    [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
    [-20, -40, -60, -80]
  ]]];

  const dataFormats: DataFormat[] =
      [undefined, 'channelsFirst', 'channelsLast'];
  const paddingModes: PaddingMode[] = [undefined, 'same', 'valid'];
  const stridesArray = [1, 2];
  const depthMultipliers = [1, 2];

  for (const dataFormat of dataFormats) {
    for (const paddingMode of paddingModes) {
      for (const stride of stridesArray) {
        for (const depthMultiplier of depthMultipliers) {
          const testTitle = `stride=${stride}, ${paddingMode}, ` +
              `${dataFormat}, depthMultiplier=${depthMultiplier}`;
          it(testTitle, () => {
            let x: Tensor = tensor4d(x4by4Data, [1, 1, 4, 4]);
            if (dataFormat !== 'channelsFirst') {
              x = tfc.transpose(x, [0, 2, 3, 1]);  // NCHW -> NHWC.
            }

            let kernel: Tensor4D;
            if (depthMultiplier === 1) {
              kernel = tensor4d([1, 0, 0, -1], [2, 2, 1, 1]);
            } else if (depthMultiplier === 2) {
              // Two kernels of the same absolute values but opposite signs:
              //   [[1, 0], [0, -1]] and [[-1, 0], [0, 1]].
              kernel = tensor4d([1, -1, 0, 0, 0, 0, -1, 1], [2, 2, 1, 2]);
            }
            const y = depthwiseConv2d(
                x, kernel, [stride, stride], 'valid', dataFormat);

            let yExpected: Tensor;
            if (stride === 1) {
              if (depthMultiplier === 1) {
                yExpected = tensor4d(
                    [[[[-30, -30, -30], [50, 90, 130], [30, 30, 30]]]],
                    [1, 1, 3, 3]);
              } else if (depthMultiplier === 2) {
                yExpected = tensor4d(
                    [[
                      [[-30, -30, -30], [50, 90, 130], [30, 30, 30]],
                      [[30, 30, 30], [-50, -90, -130], [-30, -30, -30]]
                    ]],
                    [1, 2, 3, 3]);
              }
            } else if (stride === 2) {
              if (depthMultiplier === 1) {
                yExpected = tensor4d([[[[-30, -30], [30, 30]]]], [1, 1, 2, 2]);
              } else if (depthMultiplier === 2) {
                yExpected = tensor4d(
                    [[[[-30, -30], [30, 30]], [[30, 30], [-30, -30]]]],
                    [1, 2, 2, 2]);
              }
            }
            if (dataFormat !== 'channelsFirst') {
              yExpected = tfc.transpose(yExpected, [0, 2, 3, 1]);
            }
            expectTensorsClose(y, yExpected);
          });
        }
      }
    }
  }

  it('Non-4D kernel leads to exception', () => {
    const x = tfc.zeros([1, 1, 4, 4]);
    expect(() => depthwiseConv2d(x, tfc.zeros([1, 2, 2]), [1, 1]))
        .toThrowError(/.* is required to be 4-D, but is instead 3-D/);
  });
});

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
            const depthwiseConvLayer = tfl.layers.depthwiseConv2d(
                {dataFormat, kernelSize, depthMultiplier, padding});
            const inputShape = dataFormat === 'channelsFirst' ? [1, 8, 10, 10] :
                                                                [1, 10, 10, 8];
            const symbolicInput =
                new tfl.SymbolicTensor('float32', inputShape, null, [], null);
            const symbolicOutput =
                depthwiseConvLayer.apply(symbolicInput) as tfl.SymbolicTensor;

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
    const depthwiseConvLayer = tfl.layers.depthwiseConv2d({kernelSize: 2});
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [1, 10, 10], null, [], null);
    expect(() => depthwiseConvLayer.apply(symbolicInput))
        .toThrowError(
            /Inputs to DepthwiseConv2D should have rank 4\. Received .*/);
  });

  it('Serialization round trip', () => {
    const layer = tfl.layers.depthwiseConv2d(
        {kernelSize: 3, depthMultiplier: 4, activation: 'relu'});
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.depthwiseConv2d(tsConfig);
    const configPrime = layerPrime.getConfig();
    expect(configPrime.kernelSize).toEqual([3, 3]);
    expect(configPrime.depthMultiplier).toEqual(4);
    expect(configPrime.activation).toEqual('relu');
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
          const conv2dLayer = tfl.layers.depthwiseConv2d({
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
    const x = tfc.transpose(tensor4d(x4by4Data, [1, 1, 4, 4]), [0, 2, 3, 1]);
    const conv2dLayer = tfl.layers.depthwiseConv2d({
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

  it('missing config.kernelSize throws exception', () => {
    // tslint:disable-next-line:no-any
    expect((filters: 1) => tfl.layers.depthwiseConv2d({} as any))
        .toThrowError(/kernelSize/);
  });
  it('bad config.kernelSize throws exception', () => {
    expect(
        // tslint:disable-next-line:no-any
        () => tfl.layers.depthwiseConv2d({kernelSize: [1]} as any))
        .toThrowError(/kernelSize/);
  });
});
