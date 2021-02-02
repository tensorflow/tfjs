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
 * Unit tests for convolutional.ts.
 */

import * as tfc from '@tensorflow/tfjs-core';
import {scalar, Tensor, tensor1d, tensor3d, tensor4d, tensor5d, util} from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import {InitializerIdentifier} from '../initializers';
import {ActivationIdentifier} from '../keras_format/activation_config';
import {DataFormat, InterpolationFormat, PaddingMode, Shape} from '../keras_format/common';
import {describeMathCPU, describeMathCPUAndGPU, describeMathGPU, expectTensorsClose} from '../utils/test_utils';

import {conv1d, conv1dWithBias, conv2d, conv2dWithBiasActivation, conv3d, conv3dWithBias} from './convolutional';

describeMathCPUAndGPU('conv1dWithBias', () => {
  const xLength4Data = [10, 20, 40, 80];
  const kernelLength2Data = [1, -1];
  const biasScalarData = 2.2;
  // In the basic case, this convolves [10, 20, 40, 80] with the kernel [1, -1],
  // producing [-10, -20, -40], and adds the bias 2.2, producing
  // [-7.8, -17.8, -37.7].  The test is reproduced for either 1 or 2 output
  // channels, and several reasonable data formats.

  const outChannelsArray = [1, 2];
  const dataFormats: DataFormat[] =
      [undefined, 'channelsFirst', 'channelsLast'];
  const paddingModes: PaddingMode[] = [undefined, 'same', 'valid'];
  const stride = 1;

  for (const outChannels of outChannelsArray) {
    for (const dataFormat of dataFormats) {
      for (const paddingMode of paddingModes) {
        const testTitle = `outChannels=${outChannels}, stride=${stride}, ` +
            `${paddingMode}, ${dataFormat}`;
        it(testTitle, () => {
          let x: Tensor = tensor3d(xLength4Data, [1, 4, 1]);
          if (dataFormat === 'channelsFirst') {
            x = tfc.transpose(x, [0, 2, 1]);  // NWC -> NCW.
          }

          let kernelData: number[] = [];
          let biasData: number[] = [];
          for (let i = 0; i < outChannels; ++i) {
            kernelData = kernelData.concat(kernelLength2Data);
            biasData = biasData.concat([biasScalarData + i]);
          }
          const kernel = tfc.transpose(
              tensor3d(kernelData, [1, outChannels, 2]), [2, 0, 1]);
          const bias = tensor1d(biasData);

          const y =
              conv1dWithBias(x, kernel, bias, stride, paddingMode, dataFormat);

          let yExpectedShape: [number, number, number];
          let yExpectedData: number[];
          if (paddingMode === 'valid' || paddingMode === undefined) {
            if (outChannels === 1) {
              yExpectedShape = [1, 3, 1];
              yExpectedData = [-7.8, -17.8, -37.8];
            } else if (outChannels === 2) {
              yExpectedShape = [1, 3, 2];
              yExpectedData = [-7.8, -6.8, -17.8, -16.8, -37.8, -36.8];
            }
          } else if (paddingMode === 'same') {
            if (outChannels === 1) {
              yExpectedShape = [1, 4, 1];
              yExpectedData = [-7.8, -17.8, -37.8, 82.2];
            } else if (outChannels === 2) {
              yExpectedShape = [1, 4, 2];
              yExpectedData =
                  [-7.8, -6.8, -17.8, -16.8, -37.8, -36.8, 82.2, 83.2];
            }
          }
          expectTensorsClose(y, tensor3d(yExpectedData, yExpectedShape));
        });
      }
    }
  }
});

describeMathCPUAndGPU('conv1d', () => {
  const xLength4Data = [10, 20, 40, 80];
  const kernelLength2Data = [1, -1];

  const outChannels = 2;
  const dataFormat = 'channelsLast';
  const paddingMode = 'valid';
  const strides = [2, 1];
  const dilations = [1, 2];
  const expectations = [[-10, -10, -40, -40], [-30, -30, -60, -60]];

  for (let i = 0; i < strides.length; ++i) {
    const stride = strides[i];
    const dilationRate = dilations[i];
    const expectation = expectations[i];
    const testTitle = `outChannels=${outChannels}, stride=${stride}, ` +
        `${paddingMode}, dilationRate=${dilationRate}, ${dataFormat}`;
    it(testTitle, () => {
      const x = tensor3d(xLength4Data, [1, 4, 1]);
      let kernelData: number[] = [];
      for (let i = 0; i < outChannels; ++i) {
        kernelData = kernelData.concat(kernelLength2Data);
      }
      const kernel =
          tfc.transpose(tensor3d(kernelData, [1, outChannels, 2]), [2, 0, 1]);
      const y =
          conv1d(x, kernel, stride, paddingMode, dataFormat, dilationRate);
      expectTensorsClose(y, tensor3d(expectation, [1, 2, 2]));
    });
  }
});

describeMathCPUAndGPU('conv2d', () => {
  const x4by4Data = [[[
    [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
    [-20, -40, -60, -80]
  ]]];
  const kernel2by2Data = [1, 0, 0, -1];

  const dataFormats: DataFormat[] =
      [undefined, 'channelsFirst', 'channelsLast'];
  const paddingModes: PaddingMode[] = [undefined, 'same', 'valid'];
  const stridesArray = [1, 2];

  for (const dataFormat of dataFormats) {
    for (const paddingMode of paddingModes) {
      for (const stride of stridesArray) {
        const testTitle = `stride=${stride}, ${paddingMode}, ` +
            `${dataFormat}`;
        it(testTitle, () => {
          let x: Tensor = tensor4d(x4by4Data, [1, 1, 4, 4]);
          if (dataFormat !== 'channelsFirst') {
            x = tfc.transpose(x, [0, 2, 3, 1]);  // NCHW -> NHWC.
          }
          const kernel = tensor4d(kernel2by2Data, [2, 2, 1, 1]);
          const y = conv2d(x, kernel, [stride, stride], 'valid', dataFormat);

          let yExpected: Tensor;
          if (stride === 1) {
            yExpected = tensor4d(
                [[[[-30, -30, -30], [50, 90, 130], [30, 30, 30]]]],
                [1, 1, 3, 3]);
          } else if (stride === 2) {
            yExpected = tensor4d([[[[-30, -30], [30, 30]]]], [1, 1, 2, 2]);
          }
          if (dataFormat !== 'channelsFirst') {
            yExpected = tfc.transpose(yExpected, [0, 2, 3, 1]);
          }
          expectTensorsClose(y, yExpected);
        });
      }
    }
  }

  it('Invalid filters leads to Error', () => {
    expect(() => tfl.layers.conv2d({filters: 2.5, kernelSize: 3}))
        .toThrowError(/filters.*positive integer.*2\.5\.$/);
  });
});

describeMathCPUAndGPU('conv2dWithBiasActivation', () => {
  const x4by4Data = [[[
    [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
    [-20, -40, -60, -80]
  ]]];
  const kernel2by2Data = [1, 0, 0, -1];
  const biasScalarData = [2.2];

  const outChannelsArray = [2, 3];
  const dataFormats: DataFormat[] =
      [undefined, 'channelsFirst', 'channelsLast'];
  const paddingModes: PaddingMode[] = [undefined, 'same', 'valid'];
  const stridesArray = [1, 2];

  for (const outChannels of outChannelsArray) {
    for (const dataFormat of dataFormats) {
      for (const paddingMode of paddingModes) {
        for (const stride of stridesArray) {
          const testTitle = `outChannels=${outChannels}, stride=${stride}, ` +
              `${paddingMode}, ${dataFormat}`;
          it(testTitle, () => {
            let x: Tensor = tensor4d(x4by4Data, [1, 1, 4, 4]);
            if (dataFormat !== 'channelsFirst') {
              x = tfc.transpose(x, [0, 2, 3, 1]);  // NCHW -> NHWC.
            }

            let kernelData: number[] = [];
            let biasData: number[] = [];
            for (let i = 0; i < outChannels; ++i) {
              kernelData = kernelData.concat(kernel2by2Data);
              biasData = biasData.concat(biasScalarData);
            }
            const kernel = tfc.transpose(
                tensor4d(kernelData, [outChannels, 2, 2, 1]), [1, 2, 3, 0]);
            const bias = tensor1d(biasData);

            const y = conv2dWithBiasActivation(
                x, kernel, bias, [stride, stride], 'valid', dataFormat);

            let yExpectedShape: [number, number, number, number];
            let yExpectedDataPerChannel: number[];
            if (stride === 1) {
              yExpectedShape = [1, outChannels, 3, 3];
              yExpectedDataPerChannel =
                  [-30, -30, -30, 50, 90, 130, 30, 30, 30];
            } else if (stride === 2) {
              yExpectedShape = [1, outChannels, 2, 2];
              yExpectedDataPerChannel = [-30, -30, 30, 30];
            }
            for (let i = 0; i < yExpectedDataPerChannel.length; ++i) {
              yExpectedDataPerChannel[i] += biasScalarData[0];
            }
            let yExpectedData: number[] = [];
            for (let i = 0; i < outChannels; ++i) {
              yExpectedData = yExpectedData.concat(yExpectedDataPerChannel);
            }
            let yExpected: Tensor = tensor4d(yExpectedData, yExpectedShape);
            if (dataFormat !== 'channelsFirst') {
              yExpected = tfc.transpose(yExpected, [0, 2, 3, 1]);
            }
            expectTensorsClose(y, yExpected);
          });
        }
      }
    }
  }
});

describeMathCPU('Conv2D Layers: Symbolic', () => {
  const filtersArray = [1, 64];
  const paddingModes: PaddingMode[] = [undefined, 'valid', 'same'];
  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];
  const kernelSizes = [[2, 2], [3, 4]];
  // In this test suite, `undefined` means strides is the same as kernelSize.
  const stridesArray = [undefined, 1];

  for (const filters of filtersArray) {
    for (const padding of paddingModes) {
      for (const dataFormat of dataFormats) {
        for (const kernelSize of kernelSizes) {
          for (const stride of stridesArray) {
            const strides = stride || kernelSize;
            const testTitle = `filters=${filters}, kernelSize=${
                                  JSON.stringify(kernelSize)}, ` +
                `strides=${JSON.stringify(strides)}, ` +
                `${dataFormat}, ${padding}`;
            it(testTitle, () => {
              const inputShape = dataFormat === 'channelsFirst' ?
                  [2, 16, 11, 9] :
                  [2, 11, 9, 16];
              const symbolicInput =
                  new tfl.SymbolicTensor('float32', inputShape, null, [], null);

              const conv2dLayer = tfl.layers.conv2d({
                filters,
                kernelSize,
                strides,
                padding,
                dataFormat,
              });

              const output =
                  conv2dLayer.apply(symbolicInput) as tfl.SymbolicTensor;

              let outputRows: number;
              let outputCols: number;
              if (stride === undefined) {  // Same strides as kernelSize.
                outputRows = kernelSize[0] === 2 ? 5 : 3;
                if (padding === 'same') {
                  outputRows++;
                }
                outputCols = kernelSize[1] === 2 ? 4 : 2;
                if (padding === 'same') {
                  outputCols++;
                }
              } else {  // strides: 1.
                outputRows = kernelSize[0] === 2 ? 10 : 9;
                if (padding === 'same') {
                  outputRows += kernelSize[0] - 1;
                }
                outputCols = kernelSize[1] === 2 ? 8 : 6;
                if (padding === 'same') {
                  outputCols += kernelSize[1] - 1;
                }
              }
              let expectedShape: [number, number, number, number];
              if (dataFormat === 'channelsFirst') {
                expectedShape = [2, filters, outputRows, outputCols];
              } else {
                expectedShape = [2, outputRows, outputCols, filters];
              }

              expect(output.shape).toEqual(expectedShape);
              expect(output.dtype).toEqual(symbolicInput.dtype);
            });
          }
        }
      }
    }
  }

  it('missing config.kernelSize throws exception', () => {
    // tslint:disable-next-line:no-any
    expect((filters: 1) => tfl.layers.conv2d({filters: 1} as any))
        .toThrowError(/kernelSize/);
  });
  it('bad config.kernelSize shape throws exception', () => {
    expect(() => tfl.layers.conv2d({filters: 1, kernelSize: [1, 1, 1]}))
        .toThrowError(
            /kernelSize argument must be an integer or tuple of 2 integers/);
  });
  it('bad config.kernelSize shape throws exception', () => {
    expect(() => tfl.layers.conv2d({filters: 1, kernelSize: [1, 1, 1, 1]}))
        .toThrowError(
            /kernelSize to be number or number\[\] with length 1, 2, or 3/);
  });
  it('missing config.filters throws exception', () => {
    // tslint:disable-next-line:no-any
    expect(() => tfl.layers.conv2d({kernelSize: 1} as any))
        .toThrowError(/filters to be a 'number' > 0/);
  });
  it('bad config.filters value throws exception', () => {
    // tslint:disable-next-line:no-any
    expect(() => tfl.layers.conv2d({kernelSize: 1, filters: 0} as any))
        .toThrowError(/filters to be a 'number' > 0/);
  });
});

describeMathCPUAndGPU('Conv2D Layer: Tensor', () => {
  const x4by4Data = [[[
    [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
    [-20, -40, -60, -80]
  ]]];

  const useBiases = [false, true];
  const biasInitializers: InitializerIdentifier[] = ['zeros', 'ones'];
  const activations: ActivationIdentifier[] = [null, 'linear', 'relu'];

  for (const useBias of useBiases) {
    for (const biasInitializer of biasInitializers) {
      for (const activation of activations) {
        const testTitle =
            `useBias=${useBias}, biasInitializer=${biasInitializer}, ` +
            `activation=${activation}`;
        it(testTitle, () => {
          const x = tensor4d(x4by4Data, [1, 1, 4, 4]);
          const conv2dLayer = tfl.layers.conv2d({
            filters: 1,
            kernelSize: [2, 2],
            strides: [2, 2],
            dataFormat: 'channelsFirst',
            useBias,
            kernelInitializer: 'ones',
            biasInitializer,
            activation
          });
          const y = conv2dLayer.apply(x) as Tensor;

          let yExpectedData = [100, 260, -100, -260];
          if (useBias && biasInitializer === 'ones') {
            yExpectedData = yExpectedData.map(element => element + 1);
          }
          if (activation === 'relu') {
            yExpectedData =
                yExpectedData.map(element => element >= 0 ? element : 0);
          }
          const yExpected = tensor4d(yExpectedData, [1, 1, 2, 2]);
          expectTensorsClose(y, yExpected);
        });
      }
    }
  }

  it('CHANNEL_LAST', () => {
    // Convert input to CHANNEL_LAST.
    const x = tfc.transpose(tensor4d(x4by4Data, [1, 1, 4, 4]), [0, 2, 3, 1]);
    const conv2dLayer = tfl.layers.conv2d({
      filters: 1,
      kernelSize: [2, 2],
      strides: [2, 2],
      dataFormat: 'channelsLast',
      useBias: false,
      kernelInitializer: 'ones',
      activation: 'linear'
    });
    const y = conv2dLayer.apply(x) as Tensor;
    const yExpected = tensor4d([100, 260, -100, -260], [1, 2, 2, 1]);
    expectTensorsClose(y, yExpected);
  });

  const dilationRateValues: Array<number|[number, number]> = [2, [2, 2]];
  for (const dilationRate of dilationRateValues) {
    it(`CHANNEL_LAST, dilationRate=${dilationRate}`, () => {
      const x = tensor4d(
          [[
            [
              [0.89240986], [0.54892443], [0.24670805], [0.03983783],
              [0.56602233]
            ],

            [
              [0.21421895], [0.58529864], [0.60060781], [0.66895784],
              [0.08855761]
            ],

            [
              [0.56657235], [0.25803428], [0.17971111], [0.65166403],
              [0.70492866]
            ],

            [
              [0.46641512], [0.05765411], [0.52517211], [0.62557303],
              [0.30612501]
            ],

            [
              [0.8406994], [0.56932724], [0.96028134], [0.34666753],
              [0.04458038]
            ]
          ]],
          [1, 5, 5, 1]);
      const conv2dLayer = tfl.layers.conv2d({
        filters: 1,
        kernelSize: [2, 2],
        strides: 1,
        dataFormat: 'channelsLast',
        useBias: false,
        kernelInitializer: 'ones',
        activation: 'linear',
        dilationRate
      });
      const y = conv2dLayer.apply(x) as Tensor;
      const yExpected = tensor4d(
          [[
            [[1.8854014], [1.4984605], [1.6973702]],

            [[1.8064139], [1.9374835], [1.5204625]],

            [[2.547264], [1.8256931], [1.8895016]]
          ]],
          [1, 3, 3, 1]);
      expectTensorsClose(y, yExpected);
    });
  }

  const explicitDefaultDilations: Array<number|[number, number]> = [1, [1, 1]];
  for (const explicitDefaultDilation of explicitDefaultDilations) {
    const testTitle = 'Explicit default dilation rate: ' +
        JSON.stringify(explicitDefaultDilation);
    it(testTitle, () => {
      const conv2dLayer = tfl.layers.conv2d({
        filters: 1,
        kernelSize: [2, 2],
        strides: [2, 2],
        dataFormat: 'channelsFirst',
        useBias: false,
        kernelInitializer: 'ones',
        dilationRate: explicitDefaultDilation
      });
      const x = tensor4d(x4by4Data, [1, 1, 4, 4]);
      const y = conv2dLayer.apply(x) as Tensor;
      const yExpected = tensor4d([100, 260, -100, -260], [1, 1, 2, 2]);
      expectTensorsClose(y, yExpected);
    });
  }
});

describeMathCPUAndGPU('conv3d', () => {
  // # The following TensorFlow Python code is used to verify the results of
  // # the 3D convolutions.
  //
  // ```python
  // import numpy as np
  // import tensorflow as tf
  // tf.enable_eager_execution()
  //
  // x = np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4, 1)
  // # Shape of kernel is (height, width, depth, input_ch, output_ch)
  // kernel = np.array([
  //     1, 1, 1, -1, -1, 1, 1, 1
  //   ], dtype=np.float32).reshape(2, 2, 2, 1, 1)
  //
  // # Strides (1, 1, 1)
  // outputs = tf.nn.conv3d(
  //     input=x, filter=kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
  // outputs.numpy().flatten()
  // # Output:
  // #   array([ 42.,  46.,  50.,  58.,  62.,  66.,  74.,  78.,  82., 106.,
  // #           110., 114., 122., 126., 130., 138., 142., 146., 170., 174.,
  // #           178., 186., 190., 194., 202., 206., 210.], dtype=float32)
  // outputs.numpy().shape  # (1, 3, 3, 3, 1)
  //
  // # Strides (2, 2, 2)
  // outputs = tf.nn.conv3d(
  //     input=x, filter=kernel, strides=[1, 2, 2, 2, 1], padding='VALID')
  // outputs.numpy().flatten()
  // # Output:
  // #   array([ 42.,  50.,  74.,  82., 170., 178., 202., 210.], dtype=float32)
  // outputs.numpy().shape  # (1, 2, 2, 2, 1)
  // ```

  const x4by4by4Data = [
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
  ];
  const kernel2by2by2Data = [1, 1, 1, -1, -1, 1, 1, 1];

  const dataFormats: DataFormat[] =
      [undefined, 'channelsFirst', 'channelsLast'];
  const paddingModes: PaddingMode[] = [undefined, 'same', 'valid'];
  const stridesArray = [1, 2];

  for (const dataFormat of dataFormats) {
    for (const paddingMode of paddingModes) {
      for (const stride of stridesArray) {
        const testTitle = `stride=${stride}, ${paddingMode}, ` +
            `${dataFormat}`;
        it(testTitle, () => {
          let x = tensor5d(x4by4by4Data, [1, 1, 4, 4, 4]);
          if (dataFormat !== 'channelsFirst') {
            x = tfc.transpose(x, [0, 2, 3, 4, 1]);  // NCDHW -> NDHWC.
          }
          const kernel = tensor5d(kernel2by2by2Data, [2, 2, 2, 1, 1]);
          const y =
              conv3d(x, kernel, [stride, stride, stride], 'valid', dataFormat);

          let yExpected: Tensor;
          if (stride === 1) {
            // See validation Python code in comment above.
            yExpected = tensor5d(
                [
                  42.,  46.,  50.,  58.,  62.,  66.,  74.,  78.,  82.,
                  106., 110., 114., 122., 126., 130., 138., 142., 146.,
                  170., 174., 178., 186., 190., 194., 202., 206., 210.
                ],
                [1, 1, 3, 3, 3]);
          } else if (stride === 2) {
            // See validation Python code in comment above.
            yExpected = tensor5d(
                [42., 50., 74., 82., 170., 178., 202., 210.], [1, 1, 2, 2, 2]);
          }
          if (dataFormat !== 'channelsFirst') {
            yExpected = tfc.transpose(yExpected, [0, 2, 3, 4, 1]);
          }
          expectTensorsClose(y, yExpected);
        });
      }
    }
  }

  const biasScalarData = [2.2];
  const outChannelsArray = [2, 3];

  for (const outChannels of outChannelsArray) {
    for (const dataFormat of dataFormats) {
      for (const paddingMode of paddingModes) {
        for (const stride of stridesArray) {
          const testTitle = `outChannels=${outChannels}, stride=${stride}, ` +
              `${paddingMode}, ${dataFormat}`;
          it(testTitle, () => {
            let x: Tensor = tensor5d(x4by4by4Data, [1, 1, 4, 4, 4]);
            if (dataFormat !== 'channelsFirst') {
              x = tfc.transpose(x, [0, 2, 3, 4, 1]);  // NCDHW -> NDHWC.
            }

            let kernelData: number[] = [];
            let biasData: number[] = [];
            for (let i = 0; i < outChannels; ++i) {
              kernelData = kernelData.concat(kernel2by2by2Data);
              biasData = biasData.concat(biasScalarData);
            }
            const kernel = tfc.transpose(
                tensor5d(kernelData, [outChannels, 2, 2, 2, 1]),
                [1, 2, 3, 4, 0]);
            const bias = tensor1d(biasData);

            const y = conv3dWithBias(
                x, kernel, bias, [stride, stride, stride], 'valid', dataFormat);

            let yExpectedShape: [number, number, number, number, number];
            let yExpectedDataPerChannel: number[];
            if (stride === 1) {
              yExpectedShape = [1, outChannels, 3, 3, 3];
              // See validation Python code above.
              yExpectedDataPerChannel = [
                42.,  46.,  50.,  58.,  62.,  66.,  74.,  78.,  82.,
                106., 110., 114., 122., 126., 130., 138., 142., 146.,
                170., 174., 178., 186., 190., 194., 202., 206., 210.
              ];
            } else if (stride === 2) {
              yExpectedShape = [1, outChannels, 2, 2, 2];
              // See validation Python code above.
              yExpectedDataPerChannel =
                  [42., 50., 74., 82., 170., 178., 202., 210.];
            }
            for (let i = 0; i < yExpectedDataPerChannel.length; ++i) {
              yExpectedDataPerChannel[i] += biasScalarData[0];
            }
            let yExpectedData: number[] = [];
            for (let i = 0; i < outChannels; ++i) {
              yExpectedData = yExpectedData.concat(yExpectedDataPerChannel);
            }
            let yExpected: Tensor = tensor5d(yExpectedData, yExpectedShape);
            if (dataFormat !== 'channelsFirst') {
              yExpected = tfc.transpose(yExpected, [0, 2, 3, 4, 1]);
            }
            expectTensorsClose(y, yExpected);
          });
        }
      }
    }
  }
});

describeMathCPU('Conv3D Layers: Symbolic', () => {
  const filtersArray = [1, 64];
  const paddingModes: PaddingMode[] = [undefined, 'valid', 'same'];
  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];
  const kernelSizes = [[2, 2, 2], [3, 4, 4]];
  // In this test suite, `undefined` means strides is the same as kernelSize.
  const stridesArray = [undefined, 1];

  for (const filters of filtersArray) {
    for (const padding of paddingModes) {
      for (const dataFormat of dataFormats) {
        for (const kernelSize of kernelSizes) {
          for (const stride of stridesArray) {
            const strides = stride || kernelSize;
            const testTitle = `filters=${filters}, kernelSize=${
                                  JSON.stringify(kernelSize)}, ` +
                `strides=${JSON.stringify(strides)}, ` +
                `${dataFormat}, ${padding}`;
            it(testTitle, () => {
              const inputShape = dataFormat === 'channelsFirst' ?
                  [2, 16, 11, 9, 9] :
                  [2, 11, 9, 9, 16];
              const symbolicInput =
                  new tfl.SymbolicTensor('float32', inputShape, null, [], null);

              const conv3dLayer = tfl.layers.conv3d({
                filters,
                kernelSize,
                strides,
                padding,
                dataFormat,
              });

              const output =
                  conv3dLayer.apply(symbolicInput) as tfl.SymbolicTensor;

              let outputRows: number;
              let outputCols: number;
              let outputDepth: number;
              if (stride === undefined) {  // Same strides as kernelSize.
                outputRows = kernelSize[0] === 2 ? 5 : 3;
                if (padding === 'same') {
                  outputRows++;
                }
                outputCols = kernelSize[1] === 2 ? 4 : 2;
                if (padding === 'same') {
                  outputCols++;
                }
                outputDepth = kernelSize[2] === 2 ? 4 : 2;
                if (padding === 'same') {
                  outputDepth++;
                }
              } else {  // strides: 1.
                outputRows = kernelSize[0] === 2 ? 10 : 9;
                if (padding === 'same') {
                  outputRows += kernelSize[0] - 1;
                }
                outputCols = kernelSize[1] === 2 ? 8 : 6;
                if (padding === 'same') {
                  outputCols += kernelSize[1] - 1;
                }
                outputDepth = kernelSize[2] === 2 ? 8 : 6;
                if (padding === 'same') {
                  outputDepth += kernelSize[1] - 1;
                }
              }
              let expectedShape: [number, number, number, number, number];
              if (dataFormat === 'channelsFirst') {
                expectedShape =
                    [2, filters, outputRows, outputCols, outputDepth];
              } else {
                expectedShape =
                    [2, outputRows, outputCols, outputDepth, filters];
              }

              expect(output.shape).toEqual(expectedShape);
              expect(output.dtype).toEqual(symbolicInput.dtype);
            });
          }
        }
      }
    }
  }

  it('Invalid filters leads to Error', () => {
    expect(() => tfl.layers.conv3d({filters: 2.5, kernelSize: 3}))
        .toThrowError(/filters.*positive integer.*2\.5\.$/);
  });
  it('missing config.kernelSize throws exception', () => {
    // tslint:disable-next-line:no-any
    expect((filters: 1) => tfl.layers.conv3d({filters: 1} as any))
        .toThrowError(/kernelSize/);
  });
  it('scalar config.kernelSize shape does not throw exception', () => {
    expect(() => tfl.layers.conv3d({filters: 1, kernelSize: 1}))
        .not.toThrowError();
  });
  it('bad config.kernelSize shape throws exception', () => {
    expect(() => tfl.layers.conv3d({filters: 1, kernelSize: [1, 1]}))
        .toThrowError(
            /kernelSize argument must be an integer or tuple of 3 integers/);
  });
  it('bad config.kernelSize shape throws exception', () => {
    expect(() => tfl.layers.conv2d({filters: 1, kernelSize: [1, 1, 1, 1]}))
        .toThrowError(
            /kernelSize to be number or number\[\] with length 1, 2, or 3/);
  });
  it('missing config.filters throws exception', () => {
    // tslint:disable-next-line:no-any
    expect(() => tfl.layers.conv3d({kernelSize: 1} as any))
        .toThrowError(/filters to be a 'number' > 0/);
  });
  it('bad config.filters value throws exception', () => {
    // tslint:disable-next-line:no-any
    expect(() => tfl.layers.conv3d({kernelSize: 1, filters: 0} as any))
        .toThrowError(/filters to be a 'number' > 0/);
  });
});

describeMathCPUAndGPU('Conv3D Layer: Tensor', () => {
  // # The following TensorFlow Python code is used to verify the results of
  // # the 3D convolution layer.
  //
  // ```python
  // import numpy as np
  // import tensorflow as tf
  // tf.enable_eager_execution()
  //
  // x = np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4, 1)
  // layer = tf.keras.layers.Conv3D(
  //     filters=1, kernel_size=(2, 2, 2), strides=(2, 2, 2),
  //     use_bias=True, kernel_initializer='ones', bias_initializer='zeros')
  // outputs = layer(x)
  // outputs.numpy().flatten()
  // # Output:
  // #  array([ 84., 100., 148., 164., 340., 356., 404., 420.], dtype=float32)
  // outputs.numpy().shape
  // # Output:
  // #  (1, 2, 2, 2, 1)
  // # TensorFlow Keras (Python) does not have a CPU implementation for channels
  // # first yet, so the validation code uses channels last. Axes 1 and 4 would
  // # be swapped to get channels first.
  // ```

  const x4by4by4Data = [
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
  ];

  const useBiases = [false, true];
  const biasInitializers: InitializerIdentifier[] = ['zeros', 'ones'];
  const activations: ActivationIdentifier[] = [null, 'linear', 'relu'];

  for (const useBias of useBiases) {
    for (const biasInitializer of biasInitializers) {
      for (const activation of activations) {
        const testTitle =
            `useBias=${useBias}, biasInitializer=${biasInitializer}, ` +
            `activation=${activation}`;
        it(testTitle, () => {
          const x = tensor5d(x4by4by4Data, [1, 1, 4, 4, 4]);
          const conv3dLayer = tfl.layers.conv3d({
            filters: 1,
            kernelSize: [2, 2, 2],
            strides: [2, 2, 2],
            dataFormat: 'channelsFirst',
            useBias,
            kernelInitializer: 'ones',
            biasInitializer,
            activation
          });
          const y = conv3dLayer.apply(x) as Tensor;
          // See validation Python code above.
          let yExpectedData = [84., 100., 148., 164., 340., 356., 404., 420.];
          if (useBias && biasInitializer === 'ones') {
            yExpectedData = yExpectedData.map(element => element + 1);
          }
          if (activation === 'relu') {
            yExpectedData =
                yExpectedData.map(element => element >= 0 ? element : 0);
          }
          const yExpected = tensor5d(yExpectedData, [1, 1, 2, 2, 2]);
          expectTensorsClose(y, yExpected);
        });
      }
    }
  }

  it('CHANNEL_LAST', () => {
    // Convert input to CHANNEL_LAST.
    const x =
        tfc.transpose(tensor5d(x4by4by4Data, [1, 1, 4, 4, 4]), [0, 2, 3, 4, 1]);
    const conv3dLayer = tfl.layers.conv3d({
      filters: 1,
      kernelSize: [2, 2, 2],
      strides: [2, 2, 2],
      dataFormat: 'channelsLast',
      useBias: false,
      kernelInitializer: 'ones',
      activation: 'linear'
    });
    const y = conv3dLayer.apply(x) as Tensor;
    // See validation Python code above.
    const yExpected = tensor5d(
        [84., 100., 148., 164., 340., 356., 404., 420.], [1, 2, 2, 2, 1]);
    expectTensorsClose(y, yExpected);
  });

  // # The following TensorFlow Python code is used to verify the results of
  // # the 3D convolution layer with dilation.
  //
  // ```python
  // import numpy as np
  // import tensorflow as tf
  // tf.enable_eager_execution()
  //
  // np.random.seed(42)
  // x = np.random.rand(1, 4, 4, 4, 1)
  // layer = tf.keras.layers.Conv3D(
  //     filters=1, kernel_size=(2, 2, 2), strides=1, dilation_rate=2,
  //     use_bias=False, kernel_initializer='ones', activation='linear')
  // outputs = layer(x)
  // outputs.numpy().flatten()
  // # Output:
  // #   array([2.91534395, 6.38913542, 2.78770771, 3.1383292 , 3.04194573,
  // #          3.59669644, 4.85877068, 3.16763418])
  // outputs.numpy().shape
  // # Output:
  // #   (1, 2, 2, 2, 1)
  // ```

  const dilationRateValues: Array<number|[number, number, number]> =
      [2, [2, 2, 2]];
  for (const dilationRate of dilationRateValues) {
    it(`CHANNEL_LAST, dilationRate=${dilationRate}`, () => {
      const x = tensor5d(
          [
            0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864,
            0.15599452, 0.05808361, 0.86617615, 0.60111501, 0.70807258,
            0.02058449, 0.96990985, 0.83244264, 0.21233911, 0.18182497,
            0.18340451, 0.30424224, 0.52475643, 0.43194502, 0.29122914,
            0.61185289, 0.13949386, 0.29214465, 0.36636184, 0.45606998,
            0.78517596, 0.19967378, 0.51423444, 0.59241457, 0.04645041,
            0.60754485, 0.17052412, 0.06505159, 0.94888554, 0.96563203,
            0.80839735, 0.30461377, 0.09767211, 0.68423303, 0.44015249,
            0.12203823, 0.49517691, 0.03438852, 0.9093204,  0.25877998,
            0.66252228, 0.31171108, 0.52006802, 0.54671028, 0.18485446,
            0.96958463, 0.77513282, 0.93949894, 0.89482735, 0.59789998,
            0.92187424, 0.0884925,  0.19598286, 0.04522729, 0.32533033,
            0.38867729, 0.27134903, 0.82873751, 0.35675333
          ],
          [1, 4, 4, 4, 1]);
      const conv3dLayer = tfl.layers.conv3d({
        filters: 1,
        kernelSize: [2, 2, 2],
        strides: 1,
        dataFormat: 'channelsLast',
        useBias: false,
        kernelInitializer: 'ones',
        activation: 'linear',
        dilationRate
      });
      const y = conv3dLayer.apply(x) as Tensor;
      // See validation Python code above.
      const yExpected = tensor5d(
          [
            2.91534395, 6.38913542, 2.78770771, 3.1383292, 3.04194573,
            3.59669644, 4.85877068, 3.16763418
          ],
          [1, 2, 2, 2, 1]);
      expectTensorsClose(y, yExpected);
    });
  }

  const explicitDefaultDilations: Array<number|[number, number, number]> =
      [1, [1, 1, 1]];
  for (const explicitDefaultDilation of explicitDefaultDilations) {
    const testTitle = 'Explicit default dilation rate: ' +
        JSON.stringify(explicitDefaultDilation);
    it(testTitle, () => {
      const conv3dLayer = tfl.layers.conv3d({
        filters: 1,
        kernelSize: [2, 2, 2],
        strides: [2, 2, 2],
        dataFormat: 'channelsFirst',
        useBias: false,
        kernelInitializer: 'ones',
        dilationRate: explicitDefaultDilation
      });
      const x = tensor5d(x4by4by4Data, [1, 1, 4, 4, 4]);
      const y = conv3dLayer.apply(x) as Tensor;
      const yExpected = tensor5d(
          [84., 100., 148., 164., 340., 356., 404., 420.], [1, 1, 2, 2, 2]);
      expectTensorsClose(y, yExpected);
    });
  }
});

describeMathCPU('Conv2DTranspose: Symbolic', () => {
  const filtersArray = [1, 64];
  const paddingModes: PaddingMode[] = [undefined, 'valid', 'same'];
  const kernelSizes = [2, [2, 2], [3, 4]];
  const stridesArray = [undefined, 2];

  for (const filters of filtersArray) {
    for (const padding of paddingModes) {
      for (const kernelSize of kernelSizes) {
        for (const strides of stridesArray) {
          const testTitle = `filters=${filters}, paddingMode=${padding},` +
              `kernelSize=${JSON.stringify(kernelSize)}, strides=${strides}`;
          it(testTitle, () => {
            const inputShape = [2, 11, 9, 16];
            const x =
                new tfl.SymbolicTensor('float32', inputShape, null, [], null);

            const layer = tfl.layers.conv2dTranspose(
                {filters, kernelSize, padding, strides});
            const y = layer.apply(x) as tfl.SymbolicTensor;

            let expectedShape: [number, number, number, number];
            if (strides === undefined) {
              if (padding === 'valid' || padding === undefined) {
                if (kernelSize as number === 2 ||
                    util.arraysEqual(kernelSize as number[], [2, 2])) {
                  expectedShape = [2, 12, 10, filters];
                } else if (util.arraysEqual(kernelSize as number[], [3, 4])) {
                  expectedShape = [2, 13, 12, filters];
                }
              } else if (padding === 'same') {
                expectedShape = [2, 11, 9, filters];
              }
            } else {
              if (padding === 'valid' || padding === undefined) {
                if (kernelSize as number === 2 ||
                    util.arraysEqual(kernelSize as number[], [2, 2])) {
                  expectedShape = [2, 22, 18, filters];
                } else if (util.arraysEqual(kernelSize as number[], [3, 4])) {
                  expectedShape = [2, 23, 20, filters];
                }
              } else if (padding === 'same') {
                expectedShape = [2, 22, 18, filters];
              }
            }
            expect(y.shape).toEqual(expectedShape);
          });
        }
      }
    }
  }

  it('Correct weight names', () => {
    const x = new tfl.SymbolicTensor('float32', [1, 2, 3, 4], null, [], null);
    const layer = tfl.layers.conv2dTranspose({filters: 2, kernelSize: [3, 3]});
    layer.apply(x);  // Let the layer build first.

    expect(layer.weights.length).toEqual(2);
    expect(layer.weights[0].name.indexOf('/kernel')).toBeGreaterThan(0);
    expect(layer.weights[1].name.indexOf('/bias')).toBeGreaterThan(0);
  });
});

describeMathCPUAndGPU('Conv2DTranspose: Tensor', () => {
  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];
  const stridesArray = [2, [2, 2]];
  for (const dataFormat of dataFormats) {
    for (const strides of stridesArray) {
      const testTitle =
          `filters=8, kernelSize=[2,2], padding=valid, strides=${strides}` +
          `dataFormat=${dataFormat}`;
      it(testTitle, () => {
        const filters = 8;
        const kernelSize = [2, 2];
        const padding = 'valid';
        const strides = 2;
        const layer = tfl.layers.conv2dTranspose({
          filters,
          kernelSize,
          padding,
          strides,
          dataFormat,
          kernelInitializer: 'ones',
          biasInitializer: 'ones'
        });

        const x = tfc.ones([2, 3, 4, 2]);
        const y = layer.apply(x) as Tensor;
        if (dataFormat === 'channelsLast') {
          expectTensorsClose(y, tfc.ones([2, 6, 8, 8]).mul(scalar(3)));
        } else {
          expectTensorsClose(y, tfc.ones([2, 8, 8, 4]).mul(scalar(4)));
        }
      });
    }
  }
});

describeMathCPU('Conv1D Layers: Symbolic', () => {
  const filtersArray = [1, 4];
  const paddingModes: PaddingMode[] = [undefined, 'valid', 'same'];
  const stridesArray = [undefined, 1];

  for (const filters of filtersArray) {
    for (const padding of paddingModes) {
      for (const strides of stridesArray) {
        const testTitle = `filters=${filters}, padding=${padding}, ` +
            `strides=${strides}`;
        it(testTitle, () => {
          const inputShape = [2, 8, 3];
          const symbolicInput =
              new tfl.SymbolicTensor('float32', inputShape, null, [], null);

          const conv1dLayer = tfl.layers.conv1d({
            filters,
            kernelSize: 2,
            strides,
            padding,
            dataFormat: 'channelsLast',
          });

          const output = conv1dLayer.apply(symbolicInput) as tfl.SymbolicTensor;

          const expectedShape = [2, 7, filters];
          if (padding === 'same') {
            expectedShape[1] = 8;
          }
          expect(output.shape).toEqual(expectedShape);
          expect(output.dtype).toEqual(symbolicInput.dtype);
        });
      }
    }
  }

  it('bad config.kernelSize shape throws exception', () => {
    expect(() => tfl.layers.conv1d({filters: 1, kernelSize: [1, 1]}))
        .toThrowError(/kernelSize.*1/);
  });
});

describeMathCPUAndGPU('Conv1D Layer: Tensor', () => {
  const xLength4Data = [10, -30, -50, 70];
  // In the most basic case, applying an all-ones convolutional kernel to
  // the 1D input above gives [-20, -80, 20]. Then adding all-ones bias to
  // it gives [-19, -79, 21].

  const stridesValues = [1, 2];
  const activations: ActivationIdentifier[] = ['linear', 'relu'];
  for (const strides of stridesValues) {
    for (const activation of activations) {
      const testTitle = `useBias=true, biasInitializer=ones, ` +
          `activation=${activation}; strides=${strides}`;
      it(testTitle, () => {
        const x = tensor3d(xLength4Data, [1, 4, 1]);
        const conv1dLayer = tfl.layers.conv1d({
          filters: 1,
          kernelSize: 2,
          strides,
          dataFormat: 'channelsLast',
          useBias: true,
          kernelInitializer: 'ones',
          biasInitializer: 'ones',
          activation
        });
        const y = conv1dLayer.apply(x) as Tensor;

        let yExpectedShape: [number, number, number];
        let yExpectedData: number[];
        if (strides === 1) {
          yExpectedShape = [1, 3, 1];
          yExpectedData = [-19, -79, 21];
        } else {
          yExpectedShape = [1, 2, 1];
          yExpectedData = [-19, 21];
        }
        if (activation === 'relu') {
          yExpectedData = yExpectedData.map(x => x > 0 ? x : 0);
        }
        const yExpected = tensor3d(yExpectedData, yExpectedShape);
        expectTensorsClose(y, yExpected);
      });
    }
  }

  const dilationRates: Array<number|[number]> = [2, [2]];
  for (const dilationRate of dilationRates) {
    it(`dilationRate = ${dilationRate}`, () => {
      const x = tensor3d(
          [
            0.0024236, 0.54829558, 0.47628448, 0.2971449, 0.7984293, 0.71802861,
            0.53109141, 0.85882819
          ],
          [1, 8, 1]);
      const conv1dLayer = tfl.layers.conv1d({
        filters: 1,
        kernelSize: 2,
        strides: 1,
        useBias: true,
        kernelInitializer: 'ones',
        biasInitializer: 'ones',
        dilationRate,
      });
      const y = conv1dLayer.apply(x) as Tensor;
      const yExpected = tensor3d(
          [1.478708, 1.8454404, 2.2747138, 2.0151734, 2.3295207, 2.5768569],
          [1, 6, 1]);
      expectTensorsClose(y, yExpected);
    });
  }

  it('missing config.kernelSize throws exception', () => {
    // tslint:disable-next-line:no-any
    expect((filters: 1) => tfl.layers.conv1d({filters: 1} as any))
        .toThrowError(/required key 'kernelSize' not in config/);
  });
  it('bad config.kernelSize throws exception', () => {
    expect(() => tfl.layers.conv1d({filters: 1, kernelSize: [1, 1, 1]}))
        .toThrowError(
            /kernelSize argument must be an integer or tuple of 1 integers/);
  });
  it('missing config.filters throws exception', () => {
    // tslint:disable-next-line:no-any
    expect(() => tfl.layers.conv1d({kernelSize: 1} as any))
        .toThrowError(/filters to be a 'number' > 0/);
  });
  it('bad config.filters throws exception', () => {
    // tslint:disable-next-line:no-any
    expect(() => tfl.layers.conv1d({kernelSize: 1, filters: 0} as any))
        .toThrowError(/filters to be a 'number' > 0/);
  });
});

describeMathCPU('SeparableConv2D Layers: Symbolic', () => {
  const filtersArray = [1, 8];
  const paddingModes: PaddingMode[] = [undefined, 'valid', 'same'];
  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];
  const kernelSizes = [[2, 2], [3, 4]];
  // In this test suite, `undefined` means strides is the same as kernelSize.
  const stridesArray = [undefined, 1];
  const dilationRates = [undefined, 2];

  for (const filters of filtersArray) {
    for (const padding of paddingModes) {
      for (const dataFormat of dataFormats) {
        for (const kernelSize of kernelSizes) {
          for (const stride of stridesArray) {
            for (const dilationRate of dilationRates) {
              const strides = stride || kernelSize;
              const testTitle = `filters=${filters}, kernelSize=${
                                    JSON.stringify(kernelSize)}, ` +
                  `strides=${JSON.stringify(strides)}, ` +
                  `dataFormat=${dataFormat}, padding=${padding}, ` +
                  `dilationRate=${dilationRate}`;
              it(testTitle, () => {
                const inputShape = dataFormat === 'channelsFirst' ?
                    [2, 16, 11, 9] :
                    [2, 11, 9, 16];
                const symbolicInput = new tfl.SymbolicTensor(
                    'float32', inputShape, null, [], null);

                const layer = tfl.layers.separableConv2d({
                  filters,
                  kernelSize,
                  strides,
                  padding,
                  dataFormat,
                  dilationRate,
                });

                const output = layer.apply(symbolicInput) as tfl.SymbolicTensor;

                let outputRows: number;
                let outputCols: number;
                if (dilationRate == null) {
                  if (stride === undefined) {  // Same strides as kernelSize.
                    outputRows = kernelSize[0] === 2 ? 5 : 3;
                    if (padding === 'same') {
                      outputRows++;
                    }
                    outputCols = kernelSize[1] === 2 ? 4 : 2;
                    if (padding === 'same') {
                      outputCols++;
                    }
                  } else {  // strides: 1.
                    outputRows = kernelSize[0] === 2 ? 10 : 9;
                    if (padding === 'same') {
                      outputRows += kernelSize[0] - 1;
                    }
                    outputCols = kernelSize[1] === 2 ? 8 : 6;
                    if (padding === 'same') {
                      outputCols += kernelSize[1] - 1;
                    }
                  }
                } else {
                  if (padding === 'same') {
                    if (stride === undefined) {  // Same strides as kernelSize.
                      outputRows = kernelSize[0] === 2 ? 6 : 4;
                      outputCols = kernelSize[1] === 2 ? 5 : 3;
                    } else {  // strides: 1.
                      outputRows = 11;
                      outputCols = 9;
                    }
                  } else {
                    if (stride === undefined) {  // Same strides as kernelSize.
                      outputRows = kernelSize[0] === 2 ? 5 : 3;
                      outputCols = kernelSize[1] === 2 ? 4 : 1;
                    } else {  // strides: 1.
                      outputRows = kernelSize[0] === 2 ? 9 : 7;
                      outputCols = kernelSize[1] === 2 ? 7 : 3;
                    }
                  }
                }
                let expectedShape: [number, number, number, number];
                if (dataFormat === 'channelsFirst') {
                  expectedShape = [2, filters, outputRows, outputCols];
                } else {
                  expectedShape = [2, outputRows, outputCols, filters];
                }

                expect(output.shape).toEqual(expectedShape);
                expect(output.dtype).toEqual(symbolicInput.dtype);
              });
            }
          }
        }
      }
    }
  }

  it('Incorrect input rank throws error', () => {
    const layer = tfl.layers.separableConv2d({
      filters: 1,
      kernelSize: [2, 2],
      strides: 1,
    });
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [2, 3, 4], null, [], null);
    expect(() => layer.apply(symbolicInput)).toThrowError(/rank 4/);
  });

  it('Undefined channel axis throws error', () => {
    const layer = tfl.layers.separableConv2d({
      filters: 1,
      kernelSize: [2, 2],
      strides: 1,
    });
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [1, , 2, 3, null], null, [], null);
    expect(() => layer.apply(symbolicInput))
        .toThrowError(/channel dimension .* should be defined/);
  });
});

describeMathGPU('SeparableConv2D Layer: Tensor', () => {
  const x5by5Data = [
    1,  3,  5,  7,  9,  2,  4,   6,  8, 10, -1, -3, -5,
    -7, -9, -2, -4, -6, -8, -10, -1, 1, -1, 1,  -1
  ];

  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];
  const dilationRates: number[] = [undefined, 2];
  const useBiases = [false, true];
  const biasInitializers: InitializerIdentifier[] = ['zeros', 'ones'];
  const activations: ActivationIdentifier[] = [null, 'linear', 'relu'];

  for (const dataFormat of dataFormats) {
    for (const dilationRate of dilationRates) {
      for (const useBias of useBiases) {
        for (const biasInitializer of biasInitializers) {
          for (const activation of activations) {
            const testTitle = `dataFormat=${dataFormat}, ` +
                `dilationRate=${dilationRate}, ` +
                `useBias=${useBias}, biasInitializer=${biasInitializer}, ` +
                `activation=${activation}`;
            it(testTitle, () => {
              let x = tensor4d(x5by5Data, [1, 5, 5, 1]);
              if (dataFormat === 'channelsFirst') {
                x = tfc.transpose(x, [0, 3, 1, 2]);  // NHWC -> NCHW.
              }

              const conv2dLayer = tfl.layers.separableConv2d({
                depthMultiplier: 1,
                filters: 1,
                kernelSize: [2, 2],
                strides: 1,
                dilationRate,
                dataFormat,
                useBias,
                depthwiseInitializer: 'ones',
                pointwiseInitializer: 'ones',
                biasInitializer,
                activation
              });
              const y = conv2dLayer.apply(x) as Tensor;

              let yExpectedData: number[];
              if (dilationRate === 2) {
                yExpectedData = [0, 0, 0, 0, 0, 0, -8, -8, -16];
              } else {
                yExpectedData = [
                  10, 18, 26, 34, 2, 2, 2, 2, -10, -18, -26, -34, -6, -10, -14,
                  -18
                ];
              }
              if (useBias && biasInitializer === 'ones') {
                yExpectedData = yExpectedData.map(element => element + 1);
              }
              if (activation === 'relu') {
                yExpectedData =
                    yExpectedData.map(element => element >= 0 ? element : 0);
              }

              let yExpected = dilationRate === 2 ?
                  tensor4d(yExpectedData, [1, 3, 3, 1]) :
                  tensor4d(yExpectedData, [1, 4, 4, 1]);
              if (dataFormat === 'channelsFirst') {
                // NHWC -> NCHW.
                yExpected = tfc.transpose(yExpected, [0, 3, 1, 2]);
              }
              expectTensorsClose(y, yExpected);
            });
          }
        }
      }
    }
  }
  it('missing config.kernelSize throws exception', () => {
    // tslint:disable-next-line:no-any
    expect((filters: 1) => tfl.layers.separableConv2d({filters: 1} as any))
        .toThrowError(/kernelSize/);
  });
  it('bad config.kernelSize throws exception', () => {
    expect(
        () => tfl.layers.separableConv2d({filters: 1, kernelSize: [1, 1, 1]}))
        .toThrowError(/kernelSize/);
  });
  it('missing config.filters throws exception', () => {
    // tslint:disable-next-line:no-any
    expect(() => tfl.layers.separableConv2d({kernelSize: 1} as any))
        .toThrowError(/filters/);
  });
  it('bad config.filters throws exception', () => {
    // tslint:disable-next-line:no-any
    expect(() => tfl.layers.separableConv2d({kernelSize: 1, filters: 0} as any))
        .toThrowError(/filters/);
  });
});

describeMathCPUAndGPU('Cropping2D Layer', () => {
  it('check with undefined channels type', () => {
    const layer = tfl.layers.cropping2D({cropping: [[1, 0], [1, 0]]});
    const x = tensor4d(
        [
          [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]],
        ],
        [1, 3, 3, 1]);

    const y = tensor4d(
        [
          [[[5], [6]], [[8], [9]]],
        ],
        [1, 2, 2, 1]);

    expectTensorsClose(layer.apply(x, null) as Tensor, y);
  });

  it('check with channels last', () => {
    const layer = tfl.layers.cropping2D(
        {cropping: [[1, 1], [1, 1]], dataFormat: 'channelsLast'});
    const x = tensor4d(
        [
          [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]],
        ],
        [1, 3, 3, 1]);
    const y = tensor4d(
        [
          [[[5]]],
        ],
        [1, 1, 1, 1]);

    expectTensorsClose(layer.apply(x, null) as Tensor, y);
  });

  it('check with channels first', () => {
    const layer = tfl.layers.cropping2D(
        {cropping: [[1, 1], [1, 1]], dataFormat: 'channelsFirst'});
    const x = tensor4d(
        [
          [[[1, 2, 3], [3, 4, 5], [6, 7, 8]]],
        ],
        [1, 1, 3, 3]);
    const y = tensor4d(
        [
          [[[4]]],
        ],
        [1, 1, 1, 1]);

    expectTensorsClose(layer.apply(x, null) as Tensor, y);
  });

  it('check with non-square tensor', () => {
    const layer = tfl.layers.cropping2D(
        {cropping: [[1, 1], [1, 1]], dataFormat: 'channelsFirst'});
    const x = tensor4d(
        [
          [[[1, 2, 3, 4], [3, 4, 5, 6], [6, 7, 8, 9]]],
        ],
        [1, 1, 3, 4]);
    const y = tensor4d(
        [
          [[[4, 5]]],
        ],
        [1, 1, 1, 2]);

    expect(layer.computeOutputShape(x.shape)).toEqual(y.shape);
    expectTensorsClose(layer.apply(x, null) as Tensor, y);
  });
});

describeMathCPU('UpSampling2D Layer: Symbolic', () => {
  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];
  const interpolatinFormats: InterpolationFormat[] = ['nearest', 'bilinear'];
  const sizes = [undefined, [2, 2]];
  const undeterminedDimensionArray: string[] = [null, 'height', 'both'];

  for (const dataFormat of dataFormats) {
    for (const interpolation of interpolatinFormats) {
      for (const size of sizes) {
        for (const undeterminedDimension of undeterminedDimensionArray) {
          const testTitle = `size = ${size}, ${dataFormat}, ${interpolation}` +
              `undetermined dimension = ${
                                JSON.stringify(undeterminedDimension)}`;
          it(testTitle, () => {
            let inputShape: Shape;
            if (undeterminedDimension == null) {
              inputShape = dataFormat === 'channelsFirst' ? [2, 16, 11, 9] :
                                                            [2, 11, 9, 16];
            } else if (undeterminedDimension === 'height') {
              inputShape = dataFormat === 'channelsFirst' ? [2, 16, null, 9] :
                                                            [2, null, 9, 16];
            } else if (undeterminedDimension === 'both') {
              inputShape = dataFormat === 'channelsFirst' ?
                  [2, 16, null, null] :
                  [2, null, null, 16];
            }
            const symbolicInput =
                new tfl.SymbolicTensor('float32', inputShape, null, [], null);

            const upSampling2dLayer = tfl.layers.upSampling2d({
              size,
              dataFormat,
              interpolation,
            });

            const output =
                upSampling2dLayer.apply(symbolicInput) as tfl.SymbolicTensor;

            let outputRows: number;
            let outputCols: number;
            if (size === undefined) {
              outputRows = 2;
              outputCols = 2;
            } else {
              outputRows = size[0];
              outputCols = size[1];
            }
            let expectedShape: [number, number, number, number];
            if (undeterminedDimension == null) {
              if (dataFormat === 'channelsFirst') {
                outputRows *= inputShape[2];
                outputCols *= inputShape[3];
                expectedShape = [2, 16, outputRows, outputCols];
              } else {
                outputRows *= inputShape[1];
                outputCols *= inputShape[2];
                expectedShape = [2, outputRows, outputCols, 16];
              }
            } else if (undeterminedDimension === 'height') {
              if (dataFormat === 'channelsFirst') {
                outputCols *= inputShape[3];
                expectedShape = [2, 16, null, outputCols];
              } else {
                outputCols *= inputShape[2];
                expectedShape = [2, null, outputCols, 16];
              }
            } else if (undeterminedDimension === 'both') {
              if (dataFormat === 'channelsFirst') {
                expectedShape = [2, 16, null, null];
              } else {
                outputCols *= inputShape[2];
                expectedShape = [2, null, null, 16];
              }
            }

            expect(output.shape).toEqual(expectedShape);
          });
        }
      }
    }
  }
});

describeMathCPUAndGPU('UpSampling2D Layer', () => {
  it('check with default values', () => {
    const layer = tfl.layers.upSampling2d({});
    const x = tensor4d(
        [
          [[[1], [2]], [[3], [4]]],
        ],
        [1, 2, 2, 1]);

    const y = tensor4d(
        [
          [
            [[1], [1], [2], [2]], [[1], [1], [2], [2]], [[3], [3], [4], [4]],
            [[3], [3], [4], [4]]
          ],
        ],
        [1, 4, 4, 1]);

    expectTensorsClose(layer.apply(x, null) as Tensor, y);
  });

  it('channels last', () => {
    const layer =
        tfl.layers.upSampling2d({size: [2, 2], dataFormat: 'channelsLast'});
    const x = tensor4d(
        [
          [[[1], [2]], [[3], [4]]],
        ],
        [1, 2, 2, 1]);

    const y = tensor4d(
        [
          [
            [[1], [1], [2], [2]], [[1], [1], [2], [2]], [[3], [3], [4], [4]],
            [[3], [3], [4], [4]]
          ],
        ],
        [1, 4, 4, 1]);

    expectTensorsClose(layer.apply(x, null) as Tensor, y);
  });

  it('channels first', () => {
    const layer =
        tfl.layers.upSampling2d({size: [2, 2], dataFormat: 'channelsFirst'});
    const x = tensor4d(
        [
          [[[1, 2], [3, 4]]],
        ],
        [1, 1, 2, 2]);

    const y = tensor4d(
        [
          [[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]],
        ],
        [1, 1, 4, 4]);

    expectTensorsClose(layer.apply(x, null) as Tensor, y);
  });

  it('varying input image sizes', () => {
    const layer =
        tfl.layers.upSampling2d({size: [2, 2], dataFormat: 'channelsLast'});
    const x1 = tensor4d(
        [
          [[[1], [2]], [[3], [4]]],
        ],
        [1, 2, 2, 1]);
    const y1 = tensor4d(
        [
          [
            [[1], [1], [2], [2]], [[1], [1], [2], [2]], [[3], [3], [4], [4]],
            [[3], [3], [4], [4]]
          ],
        ],
        [1, 4, 4, 1]);
    expectTensorsClose(layer.apply(x1, null) as Tensor, y1);

    const x2 = tensor4d(
        [
          [[[1], [2]]],
        ],
        [1, 1, 2, 1]);

    const y2 = tensor4d(
        [
          [
            [[1], [1], [2], [2]],
            [[1], [1], [2], [2]],
          ],
        ],
        [1, 2, 4, 1]);
    expectTensorsClose(layer.apply(x2, null) as Tensor, y2);
  });
  it('interpolation bilinear', () => {
    const layer =
        tfl.layers.upSampling2d({size: [2, 2], interpolation: 'bilinear'});
    const x = tensor4d(
        [
          [[[1], [2]], [[3], [4]]],
        ],
        [1, 2, 2, 1]);
    const y = tensor4d(
        [
          [
            [[1], [1.5], [2], [2]], [[2], [2.5], [3], [3]],
            [[3], [3.5], [4], [4]], [[3], [3.5], [4], [4]]
          ],
        ],
        [1, 4, 4, 1]);
    expectTensorsClose(layer.apply(x, null) as Tensor, y);
  });

  it('interpolation nearest', () => {
    const layer = tfl.layers.upSampling2d(
        {size: [2, 2], dataFormat: 'channelsFirst', interpolation: 'nearest'});
    const x = tensor4d(
        [
          [[[1, 2], [3, 4]]],
        ],
        [1, 1, 2, 2]);

    const y = tensor4d(
        [
          [[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]],
        ],
        [1, 1, 4, 4]);

    expectTensorsClose(layer.apply(x, null) as Tensor, y);
  });
});
