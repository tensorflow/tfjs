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

// tslint:disable:max-line-length
import {ones, scalar, Tensor, tensor3d, Tensor4D, tensor4d, transpose, util} from '@tensorflow/tfjs-core';

import {DataFormat, PaddingMode} from '../common';
import * as tfl from '../index';
import {InitializerIdentifier} from '../initializers';
import {describeMathCPU, describeMathCPUAndGPU, describeMathGPU, expectTensorsClose} from '../utils/test_utils';
// tslint:enable:max-line-length

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
});

describeMathCPUAndGPU('Conv2D Layer: Tensor', () => {
  const x4by4Data = [[[
    [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
    [-20, -40, -60, -80]
  ]]];

  const useBiases = [false, true];
  const biasInitializers: InitializerIdentifier[] = ['zeros', 'ones'];
  const activations = [null, 'linear', 'relu'];

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
    const x = transpose(tensor4d(x4by4Data, [1, 1, 4, 4]), [0, 2, 3, 1]);
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

        const x = ones([2, 3, 4, 2]);
        const y = layer.apply(x) as Tensor;
        if (dataFormat === 'channelsLast') {
          expectTensorsClose(y, ones([2, 6, 8, 8]).mul(scalar(3)));
        } else {
          expectTensorsClose(y, ones([2, 8, 8, 4]).mul(scalar(4)));
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
});

describeMathCPUAndGPU('Conv1D Layer: Tensor', () => {
  const xLength4Data = [10, -30, -50, 70];
  // In the most basic case, applying an all-ones convolutional kernel to
  // the 1D input above gives [-20, -80, 20]. Then adding all-ones bias to
  // it gives [-19, -79, 21].

  const stridesValues = [1, 2];
  const activations = ['linear', 'relu'];
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
  const activations = [null, 'linear', 'relu'];

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
                x = transpose(x, [0, 3, 1, 2]) as Tensor4D;  // NHWC -> NCHW.
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
                yExpected = transpose(yExpected, [0, 3, 1, 2]) as
                    Tensor4D;  // NHWC -> NCHW.
              }
              expectTensorsClose(y, yExpected);
            });
          }
        }
      }
    }
  }
});

describe('Cropping2D Layer', () => {
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
});
