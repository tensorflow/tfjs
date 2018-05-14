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
 * Unit tests for pooling.ts.
 */

// tslint:disable:max-line-length
import {expandDims, Tensor, tensor2d, Tensor2D, tensor3d, tensor4d} from '@tensorflow/tfjs-core';

import {DataFormat, PaddingMode, PoolMode} from '../common';
import * as tfl from '../index';
import {SymbolicTensor} from '../types';
import {convOutputLength} from '../utils/conv_utils';
import {describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';


// tslint:enable:max-line-length

describe('Pooling Layers 1D: Symbolic', () => {
  const poolSizes = [2, 3];
  const stridesList = [null, 1, 2];
  const poolModes: PoolMode[] = ['avg', 'max'];
  const paddingModes: PaddingMode[] = [undefined, 'valid', 'same'];

  for (const poolMode of poolModes) {
    for (const paddingMode of paddingModes) {
      for (const poolSize of poolSizes) {
        for (const strides of stridesList) {
          const testTitle = `poolSize=${poolSize}, ` +
              `${paddingMode}, ${poolMode}`;
          it(testTitle, () => {
            const inputLength = 16;
            const inputNumChannels = 11;
            const inputBatchSize = 2;
            const inputShape = [inputBatchSize, inputLength, inputNumChannels];
            const symbolicInput =
                new SymbolicTensor('float32', inputShape, null, [], null);
            const poolConstructor = poolMode === 'avg' ?
                tfl.layers.averagePooling1d :
                tfl.layers.maxPooling1d;
            const poolingLayer = poolConstructor({
              poolSize,
              strides,
              padding: paddingMode,
            });
            const output = poolingLayer.apply(symbolicInput) as SymbolicTensor;
            const expectedOutputLength = convOutputLength(
                inputLength, poolSize, paddingMode,
                strides ? strides : poolSize);
            const expectedShape =
                [inputBatchSize, expectedOutputLength, inputNumChannels];

            expect(output.shape).toEqual(expectedShape);
            expect(output.dtype).toEqual(symbolicInput.dtype);
          });
        }
      }
    }
  }
});

describeMathCPUAndGPU('Pooling Layers 1D: Tensor', () => {
  const poolModes = ['avg', 'max'];
  const strides = [2, 4];
  const poolSizes = [2, 4];
  const batchSize = 2;
  for (const poolMode of poolModes) {
    for (const stride of strides) {
      for (const poolSize of poolSizes) {
        const testTitle = `stride=${stride}, ${poolMode}, ` +
            `poolSize=${poolSize}`;
        it(testTitle, () => {
          const x2by8 = tensor2d([
            [10, 30, 50, 70, 20, 40, 60, 80],
            [-10, -30, -50, -70, -20, -40, -60, -80]
          ]);
          const x2by8by1 = expandDims(x2by8, 2);
          const poolConstructor = poolMode === 'avg' ?
              tfl.layers.averagePooling1d :
              tfl.layers.maxPooling1d;
          const poolingLayer = poolConstructor({
            poolSize,
            strides: stride,
            padding: 'valid',
          });
          const output = poolingLayer.apply(x2by8by1) as Tensor;
          let outputLength: number;
          let expectedOutputVals: number[][][];
          if (poolSize === 2) {
            if (stride === 2) {
              outputLength = 4;
              if (poolMode === 'avg') {
                expectedOutputVals =
                    [[[20], [60], [30], [70]], [[-20], [-60], [-30], [-70]]];
              } else {
                expectedOutputVals =
                    [[[30], [70], [40], [80]], [[-10], [-50], [-20], [-60]]];
              }
            } else if (stride === 4) {
              outputLength = 2;
              if (poolMode === 'avg') {
                expectedOutputVals = [[[20], [30]], [[-20], [-30]]];
              } else {
                expectedOutputVals = [[[30], [40]], [[-10], [-20]]];
              }
            }
          } else if (poolSize === 4) {
            if (stride === 2) {
              outputLength = 3;
              if (poolMode === 'avg') {
                expectedOutputVals =
                    [[[40], [45], [50]], [[-40], [-45], [-50]]];
              } else {
                expectedOutputVals =
                    [[[70], [70], [80]], [[-10], [-20], [-20]]];
              }
            } else if (stride === 4) {
              outputLength = 2;
              if (poolMode === 'avg') {
                expectedOutputVals = [[[40], [50]], [[-40], [-50]]];
              } else {
                expectedOutputVals = [[[70], [80]], [[-10], [-20]]];
              }
            }
          }
          const expectedShape: [number, number, number] =
              [batchSize, outputLength, 1];
          expectTensorsClose(
              output, tensor3d(expectedOutputVals, expectedShape));
        });
      }
    }
  }
});

describe('Pooling Layers 2D: Symbolic', () => {
  const poolSizes = [2, 3];
  const poolModes = ['avg', 'max'];
  const paddingModes: PaddingMode[] = [undefined, 'valid', 'same'];
  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];
  const poolSizeIsNumberValues = [false, true];

  for (const poolMode of poolModes) {
    for (const paddingMode of paddingModes) {
      for (const dataFormat of dataFormats) {
        for (const poolSize of poolSizes) {
          for (const poolSizeIsNumber of poolSizeIsNumberValues) {
            const testTitle = `poolSize=${poolSize}, ` +
                `${dataFormat}, ${paddingMode}, ` +
                `${poolMode}, ` +
                `poollSizeIsNumber=${poolSizeIsNumber}`;
            it(testTitle, () => {
              const inputShape = dataFormat === 'channelsFirst' ?
                  [2, 16, 11, 9] :
                  [2, 11, 9, 16];
              const symbolicInput =
                  new SymbolicTensor('float32', inputShape, null, [], null);

              const poolConstructor = poolMode === 'avg' ?
                  tfl.layers.averagePooling2d :
                  tfl.layers.maxPooling2d;
              const poolingLayer = poolConstructor({
                poolSize: poolSizeIsNumber ? poolSize : [poolSize, poolSize],
                padding: paddingMode,
                dataFormat,
              });

              const output =
                  poolingLayer.apply(symbolicInput) as SymbolicTensor;

              let outputRows = poolSize === 2 ? 5 : 3;
              if (paddingMode === 'same') {
                outputRows++;
              }
              let outputCols = poolSize === 2 ? 4 : 3;
              if (paddingMode === 'same' && poolSize === 2) {
                outputCols++;
              }

              let expectedShape: [number, number, number, number];
              if (dataFormat === 'channelsFirst') {
                expectedShape = [2, 16, outputRows, outputCols];
              } else {
                expectedShape = [2, outputRows, outputCols, 16];
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

describeMathCPUAndGPU('Pooling Layers 2D: Tensor', () => {
  const x4by4Data =
      [10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80];

  const poolModes: PoolMode[] = ['avg', 'max'];
  const strides = [1, 2];
  const batchSizes = [2, 4];
  const channelsArray = [1, 3];
  for (const poolMode of poolModes) {
    for (const stride of strides) {
      for (const batchSize of batchSizes) {
        for (const channels of channelsArray) {
          const testTitle = `stride=${stride}, ${poolMode}, ` +
              `batchSize=${batchSize}, channels=${channels}`;
          it(testTitle, () => {
            let xArrayData: number[] = [];
            for (let b = 0; b < batchSize; ++b) {
              for (let c = 0; c < channels; ++c) {
                xArrayData = xArrayData.concat(x4by4Data);
              }
            }
            const x4by4 = tensor4d(xArrayData, [batchSize, channels, 4, 4]);

            const poolConstructor = poolMode === 'avg' ?
                tfl.layers.averagePooling2d :
                tfl.layers.maxPooling2d;
            const poolingLayer = poolConstructor({
              poolSize: [2, 2],
              strides: [stride, stride],
              padding: 'valid',
              dataFormat: 'channelsFirst',
            });

            const output = poolingLayer.apply(x4by4) as Tensor;

            let expectedShape: [number, number, number, number];
            let expectedOutputSlice: number[];
            if (poolMode === 'avg') {
              if (stride === 1) {
                expectedShape = [batchSize, channels, 3, 3];
                expectedOutputSlice = [25, 45, 65, 5, 5, 5, -25, -45, -65];
              } else if (stride === 2) {
                expectedShape = [batchSize, channels, 2, 2];
                expectedOutputSlice = [25, 65, -25, -65];
              }
            } else {
              if (stride === 1) {
                expectedShape = [batchSize, channels, 3, 3];
                expectedOutputSlice = [40, 60, 80, 40, 60, 80, -10, -30, -50];
              } else if (stride === 2) {
                expectedShape = [batchSize, channels, 2, 2];
                expectedOutputSlice = [40, 80, -10, -50];
              }
            }
            let expectedOutputArray: number[] = [];
            for (let b = 0; b < batchSize; ++b) {
              for (let c = 0; c < channels; ++c) {
                expectedOutputArray =
                    expectedOutputArray.concat(expectedOutputSlice);
              }
            }
            expectTensorsClose(
                output, tensor4d(expectedOutputArray, expectedShape));
          });
        }
      }
    }
  }
});

describe('1D Global pooling Layers: Symbolic', () => {
  const globalPoolingLayers =
      [tfl.layers.globalAveragePooling1d, tfl.layers.globalMaxPooling1d];

  for (const globalPoolingLayer of globalPoolingLayers) {
    const testTitle = `layer=${globalPoolingLayer.name}`;
    it(testTitle, () => {
      const inputShape = [2, 11, 9];
      const symbolicInput =
          new SymbolicTensor('float32', inputShape, null, [], null);

      const layer = globalPoolingLayer({});
      const output = layer.apply(symbolicInput) as SymbolicTensor;

      const expectedShape = [2, 9];
      expect(output.shape).toEqual(expectedShape);
      expect(output.dtype).toEqual(symbolicInput.dtype);
    });
  }
});

describeMathCPUAndGPU('1D Global Pooling Layers: Tensor', () => {
  const x3DimData = [
    [[4, -1], [0, -2], [40, -10], [0, -20]],
    [[-4, 1], [0, 2], [-40, 10], [0, 20]]
  ];
  const globalPoolingLayers =
      [tfl.layers.globalAveragePooling1d, tfl.layers.globalMaxPooling1d];
  for (const globalPoolingLayer of globalPoolingLayers) {
    const testTitle = `globalPoolingLayer=${globalPoolingLayer.name}`;
    it(testTitle, () => {
      const x = tensor3d(x3DimData, [2, 4, 2]);
      const layer = globalPoolingLayer({});
      const output = layer.apply(x) as Tensor;

      let expectedOutput: Tensor2D;
      if (globalPoolingLayer === tfl.layers.globalAveragePooling1d) {
        expectedOutput = tensor2d([[11, -8.25], [-11, 8.25]], [2, 2]);
      } else {
        expectedOutput = tensor2d([[40, -1], [0, 20]], [2, 2]);
      }
      expectTensorsClose(output, expectedOutput);
    });
  }
});


describe('2D Global pooling Layers: Symbolic', () => {
  const globalPoolingLayers =
      [tfl.layers.globalAveragePooling2d, tfl.layers.globalMaxPooling2d];
  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];

  for (const globalPoolingLayer of globalPoolingLayers) {
    for (const dataFormat of dataFormats) {
      const testTitle = `layer=${globalPoolingLayer.name}, ${dataFormat}`;
      it(testTitle, () => {
        const inputShape = [2, 16, 11, 9];
        const symbolicInput =
            new SymbolicTensor('float32', inputShape, null, [], null);

        const layer = globalPoolingLayer({dataFormat});
        const output = layer.apply(symbolicInput) as SymbolicTensor;

        const expectedShape = dataFormat === 'channelsLast' ? [2, 9] : [2, 16];
        expect(output.shape).toEqual(expectedShape);
        expect(output.dtype).toEqual(symbolicInput.dtype);
      });
    }
  }
});

describeMathCPUAndGPU('2D Global Pooling Layers: Tensor', () => {
  const x4DimData = [
    [[[4, -1], [0, -2]], [[40, -10], [0, -20]]],
    [[[4, -1], [0, -2]], [[40, -10], [0, -20]]]
  ];
  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];

  const globalPoolingLayers =
      [tfl.layers.globalAveragePooling2d, tfl.layers.globalMaxPooling2d];
  for (const globalPoolingLayer of globalPoolingLayers) {
    for (const dataFormat of dataFormats) {
      const testTitle =
          `globalPoolingLayer=${globalPoolingLayer.name}, ${dataFormat}`;
      it(testTitle, () => {
        const x = tensor4d(x4DimData, [2, 2, 2, 2]);
        const layer = globalPoolingLayer({dataFormat});
        const output = layer.apply(x) as Tensor;

        let expectedOutput: Tensor2D;
        if (globalPoolingLayer === tfl.layers.globalAveragePooling2d) {
          if (dataFormat === 'channelsFirst') {
            expectedOutput = tensor2d([[0.25, 2.5], [0.25, 2.5]], [2, 2]);
          } else {
            expectedOutput = tensor2d([[11, -8.25], [11, -8.25]], [2, 2]);
          }
        } else {
          if (dataFormat === 'channelsFirst') {
            expectedOutput = tensor2d([[4, 40], [4, 40]], [2, 2]);
          } else {
            expectedOutput = tensor2d([[40, -1], [40, -1]], [2, 2]);
          }
        }
        expectTensorsClose(output, expectedOutput);
        const config = layer.getConfig();
        expect(config.dataFormat).toEqual(dataFormat);
      });
    }
  }
});
