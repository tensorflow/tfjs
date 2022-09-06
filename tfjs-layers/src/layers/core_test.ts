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
 * Unit tests for core.ts.
 */
import {mul, ones, scalar, Tensor, tensor2d, tensor3d, tensor4d, zeros} from '@tensorflow/tfjs-core';

import * as K from '../backend/tfjs_backend';
import {SymbolicTensor} from '../engine/topology';
import * as tfl from '../index';
import {InitializerIdentifier} from '../initializers';
import {ActivationIdentifier} from '../keras_format/activation_config';
import {pyListRepeat} from '../utils/generic_utils';
import {arrayProd} from '../utils/math_utils';
import {convertPythonicToTs, convertTsToPythonic} from '../utils/serialization_utils';
import {describeMathCPU, describeMathCPUAndGPU, describeMathCPUAndWebGL2, expectTensorsClose} from '../utils/test_utils';

describe('Dropout Layer: Symbolic', () => {
  const dropoutRates = [0, 0.5];
  const symbolicInputs = [
    new tfl.SymbolicTensor('float32', [10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [null, 4], null, [], null),
  ];

  for (const rate of dropoutRates) {
    for (const symbolicInput of symbolicInputs) {
      const testTitle = `dropoutRate=${rate}; ` +
          `input shape=${JSON.stringify(symbolicInput.shape)}`;
      it(testTitle, () => {
        const dropoutLayer = tfl.layers.dropout({rate});
        const output = dropoutLayer.apply(symbolicInput) as tfl.SymbolicTensor;
        expect(output.dtype).toEqual(symbolicInput.dtype);
        expect(output.shape).toEqual(symbolicInput.shape);
        expect(output.sourceLayer).toEqual(dropoutLayer);
        expect(output.inputs).toEqual([symbolicInput]);
      });
    }
  }
});

describeMathCPUAndGPU('Dropout Layer', () => {
  describe('tensor', () => {
    const inputShape = [2, 3, 4];
    const trainingValues = [false, true];
    const dropoutRates = [0, 0.5];
    const noiseShapes = [null, inputShape, [2, 3, 1]];
    const seed = 0;

    for (const training of trainingValues) {
      for (const rate of dropoutRates) {
        for (const noiseShape of noiseShapes) {
          const testTitle = `training=${training}, dropoutRate=${rate}, ` +
              `noiseShape=${JSON.stringify(noiseShape)}`;
          it(testTitle, () => {
            const x = ones(inputShape);
            const dropoutLayer = tfl.layers.dropout({rate, noiseShape, seed});
            const y = dropoutLayer.apply(x, {training}) as Tensor;
            expect(x.dtype).toEqual(y.dtype);
            expect(x.shape).toEqual(y.shape);
            const xValue = x.dataSync();
            const yValue = y.dataSync();
            let nKept = 0;
            if (noiseShape === noiseShapes[2]) {  // customized noiseShape
              for (let i = 0; i < x.shape[0]; ++i) {
                for (let j = 0; j < x.shape[1]; ++j) {
                  const maskedValue =
                      yValue[i * x.shape[1] * x.shape[2] + j * x.shape[2]];
                  for (let k = 0; k < x.shape[2]; ++k) {
                    const indice =
                        i * x.shape[1] * x.shape[2] + j * x.shape[2] + k;
                    if (training) {
                      if (maskedValue === 0) {
                        expect(yValue[indice]).toEqual(0);
                      } else {
                        nKept++;
                        expect(yValue[indice]).toBeCloseTo(1 / (1 - rate));
                      }
                    } else {
                      nKept++;
                      expect(yValue[indice]).toEqual(1);
                    }
                  }
                }
              }
            } else {  // default noiseShape
              for (let i = 0; i < xValue.length; ++i) {
                if (yValue[i] !== 0) {
                  nKept++;
                  if (training) {
                    expect(yValue[i]).toBeCloseTo(1 / (1 - rate));
                  } else {
                    expect(yValue[i]).toBeCloseTo(1);
                  }
                }
              }
            }
            const numel = K.countParams(x);
            if (rate === 0 || !training) {
              expect(nKept).toEqual(numel);
            } else {
              expect(nKept).toBeLessThan(numel);
            }
          });
        }
      }
    }
  });

  describe('tensor with seed', () => {
    it('get specific value.', () => {
      const training = true;
      const rate = 0.5;
      const noiseShape = [2, 3, 4];
      const x = ones([2, 3, 4]);
      const seed = 23;
      const dropoutLayer = tfl.layers.dropout({rate, noiseShape, seed});
      const y = dropoutLayer.apply(x, {training}) as Tensor;
      const yValuesExpected = [
        0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 2, 0, 2, 0, 2, 2, 2, 0
      ];
      expectTensorsClose(y, tensor3d(yValuesExpected, [2, 3, 4]));
    });
  });
});

describeMathCPUAndGPU('SpatialDropout1D Layer', () => {
  for (const training of [false, true]) {
    for (const rate of [0.5, 0]) {
      it(`Forward: rate=${rate}; training=${training}`, () => {
        const layer = tfl.layers.spatialDropout1d({rate, seed: 1337});
        const xs = ones([2, 3, 4]);
        const ys = layer.apply(xs, {training}) as Tensor;
        if (!training || rate === 0) {
          expectTensorsClose(ys, xs);
        } else {
          expectTensorsClose(ys, tensor3d([
                               [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                               [[0, 2, 0, 0], [0, 2, 0, 0], [0, 2, 0, 0]]
                             ]));
        }
      });
    }
  }

  it('Incorrect input shape: Symbolic', () => {
    const layer = tfl.layers.spatialDropout1d({rate: 0.5});
    const x = new SymbolicTensor('float32', [1, 2, 3, 4], null, [], null);
    expect(() => layer.apply(x))
        .toThrowError(/.*expected ndim=3.*found ndim=4.*/);
  });

  it('Incorrect input shape: Concrete Tensor', () => {
    const layer = tfl.layers.spatialDropout1d({rate: 0.5});
    const x = ones([1, 2, 3, 4]);
    expect(() => layer.apply(x))
        .toThrowError(/.*expected ndim=3.*found ndim=4.*/);
  });

  it('Serialization round trip', () => {
    const layer = tfl.layers.spatialDropout1d({rate: 0.3, seed: 1337});
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.spatialDropout1d(tsConfig);
    expect(layerPrime.getConfig().rate).toEqual(0.3);
    expect(layerPrime.getConfig().seed).toEqual(1337);
  });
});

describeMathCPU('Dense Layer: Symbolic', () => {
  const units = 3;
  const activations: ActivationIdentifier[] =
      [null, 'linear', 'relu', 'softmax'];
  const symbolicInputs = [
    new tfl.SymbolicTensor('float32', [10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [14, 12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [null, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [null, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [null, 12, 10, 4], null, [], null),
  ];

  for (const activation of activations) {
    for (const symbolicInput of symbolicInputs) {
      it(`Generates correct symbolic output: ` +
             `activation=${activation}, ` +
             `input shape=${JSON.stringify(symbolicInput.shape)}`,
         () => {
           const denseLayer = tfl.layers.dense({units, activation});
           const output = denseLayer.apply(symbolicInput) as tfl.SymbolicTensor;

           const expectedShape = symbolicInput.shape;
           expectedShape[expectedShape.length - 1] = units;
           expect(output.shape).toEqual(expectedShape);
           expect(output.sourceLayer).toEqual(denseLayer);
           expect(output.inputs).toEqual([symbolicInput]);
         });
    }
  }

  it('2D cascade: With undetermined dimension', () => {
    const input1 = new tfl.SymbolicTensor('float32', [null, 4], null, [], null);
    const denseLayer1 = tfl.layers.dense({units: 3});
    const output1 = denseLayer1.apply(input1) as tfl.SymbolicTensor;

    const denseLayer2 = tfl.layers.dense({units: 6});
    const output2 = denseLayer2.apply(output1) as tfl.SymbolicTensor;

    expect(output1.shape).toEqual([null, 3]);
    expect(output1.sourceLayer).toEqual(denseLayer1);
    expect(output1.inputs).toEqual([input1]);
    expect(output2.shape).toEqual([null, 6]);
    expect(output2.sourceLayer).toEqual(denseLayer2);
    expect(output2.inputs).toEqual([output1]);
  });

  it('Using 1D input leads to error', () => {
    const input = new tfl.SymbolicTensor('float32', [4], null, [], null);
    const denseLayer = tfl.layers.dense({units: 3});
    expect(() => denseLayer.apply(input)).toThrowError();
  });

  it('Different rank but compatible shape works', () => {
    const denseLayer = tfl.layers.dense({units: 3});
    const input1 = new tfl.SymbolicTensor('float32', [null, 4], null, [], null);
    const input2 =
        new tfl.SymbolicTensor('float32', [null, 6, 4], null, [], null);
    const output1 = denseLayer.apply(input1) as tfl.SymbolicTensor;
    expect(output1.shape).toEqual([null, 3]);
    expect(output1.sourceLayer).toEqual(denseLayer);
    expect(output1.inputs).toEqual([input1]);

    const output2 = denseLayer.apply(input2) as tfl.SymbolicTensor;
    expect(output2.shape).toEqual([null, 6, 3]);
    expect(output2.sourceLayer).toEqual(denseLayer);
    expect(output2.inputs).toEqual([input2]);
  });

  it('2D incompatible shape leads to error', () => {
    const denseLayer = tfl.layers.dense({units: 3});
    const input1 = new tfl.SymbolicTensor('float32', [null, 4], null, [], null);
    const input2 = new tfl.SymbolicTensor('float32', [null, 5], null, [], null);
    const output1 = denseLayer.apply(input1) as tfl.SymbolicTensor;
    expect(output1.shape).toEqual([null, 3]);
    expect(output1.sourceLayer).toEqual(denseLayer);
    expect(output1.inputs).toEqual([input1]);

    expect(() => {
      // tslint:disable-next-line:no-unused-expression
      denseLayer.apply(input2);
    }).toThrowError(/incompatible with layer .* axis -1/);
  });

  it('Invalid kernelInitializer', () => {
    expect(() => {
      // tslint:disable-next-line:no-unused-expression
      tfl.layers.dense({units: 4, kernelInitializer: 'invalid_initializer!'});
    }).toThrowError(/Unknown initializer/);
  });

  it('Invalid activation', () => {
    expect(() => {
      // tslint:disable-next-line:no-unused-expression no-any
      tfl.layers.dense({units: 4, activation: 'invalid_activation!' as any});
    }).toThrowError(/Unknown activation/);
  });

  it('Invalid units leads to Error', () => {
    expect(() => tfl.layers.dense({
      units: 10.9
    })).toThrowError(/units.*positive integer.*10\.9\.$/);
    expect(() => tfl.layers.dense({
      units: 0.5
    })).toThrowError(/units.*positive integer.*0\.5\.$/);
    expect(() => tfl.layers.dense({
      units: 0
    })).toThrowError(/units.*positive integer.*0\.$/);
    expect(() => tfl.layers.dense({
      units: -2
    })).toThrowError(/units.*positive integer.*-2\.$/);
    expect(() => tfl.layers.dense({
      // tslint:disable-next-line:no-any
      units: '2' as any
    })).toThrowError(/units.*positive integer.*\"2\"\.$/);
    expect(() => tfl.layers.dense({
      // tslint:disable-next-line:no-any
      units: [] as any
    })).toThrowError('units is unexpectedly an empty array.');
  });
});

describeMathCPUAndGPU('Dense Layer: Tensor', () => {
  const units = 6;
  const useBiases = [null, false, true];
  const biasInitializers: InitializerIdentifier[] = ['zeros', 'ones'];
  const activations: ActivationIdentifier[] =
      [null, 'linear', 'relu', 'softmax'];
  const inputLastDims = [5, 8];
  // TODO(cais): Test Tensor1D, Tensor3D, Tensor4D once those are supported by
  // the backend.

  for (const useBias of useBiases) {
    for (const biasInitializer of biasInitializers) {
      for (const activation of activations) {
        for (const inputLastDim of inputLastDims) {
          it(`Call once: useBias=${useBias}, ` +
                 `biasInitializer=${biasInitializer}, ` +
                 `activation=${activation}, ` +
                 `inputLastDim=${JSON.stringify(inputLastDim)}`,
             () => {
               const input = ones([2, inputLastDim]);
               const denseLayer = tfl.layers.dense({
                 units,
                 useBias,
                 biasInitializer,
                 activation,
                 kernelInitializer: 'ones'
               });
               let expectedElementValue: number;
               if (activation === 'softmax') {
                 expectedElementValue = 1 / units;
               } else {
                 expectedElementValue = input.shape[input.shape.length - 1];
                 if (useBias !== false && biasInitializer === 'ones') {
                   expectedElementValue += 1;
                 }
               }
               const expectedShape = input.shape.slice();
               expectedShape[expectedShape.length - 1] = units;
               let expectedOutput;
               if (input.rank === 2) {
                 expectedOutput = tensor2d(
                     pyListRepeat(
                         expectedElementValue, arrayProd(expectedShape)),
                     [expectedShape[0], expectedShape[1]]);
               }
               expectTensorsClose(
                   denseLayer.apply(input, null) as Tensor, expectedOutput);
             });
        }
      }
    }
  }

  it('Calling apply again with incompatible shape leads to error', () => {
    const input1 = ones([2, 2]);  // First call.
    const input2 = ones([3, 2]);  // Okay.
    const input3 = ones([3, 3]);  // Leads to error.

    const denseLayer = tfl.layers.dense({units: 4, kernelInitializer: 'ones'});
    expectTensorsClose(
        denseLayer.apply(input1) as Tensor,
        tensor2d([2, 2, 2, 2, 2, 2, 2, 2], [2, 4]));
    expectTensorsClose(
        denseLayer.apply(input2) as Tensor,
        tensor2d([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [3, 4]));
    expect(() => denseLayer.apply(input3)).toThrowError();
  });
  it('Calling apply with compatible symbolic input after Tensor input works',
     () => {
       const concreteInput = ones([2, 2]);
       const symbolicInput =
           new tfl.SymbolicTensor('float32', [2, 2], null, [], null);
       const denseLayer =
           tfl.layers.dense({units: 4, kernelInitializer: 'ones'});

       expectTensorsClose(
           denseLayer.apply(concreteInput) as Tensor,
           tensor2d([2, 2, 2, 2, 2, 2, 2, 2], [2, 4]));

       const symbolicOuptut =
           denseLayer.apply(symbolicInput) as tfl.SymbolicTensor;
       expect(symbolicOuptut.shape).toEqual([2, 4]);
       expect(symbolicOuptut.sourceLayer).toEqual(denseLayer);
       expect(symbolicOuptut.inputs).toEqual([symbolicInput]);
     });
  it('Calling apply with incompatible symbolic input after Tensor', () => {
    const concreteInput = ones([2, 2]);
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [2, 3], null, [], null);
    const denseLayer = tfl.layers.dense({units: 4, kernelInitializer: 'ones'});

    expectTensorsClose(
        denseLayer.apply(concreteInput) as Tensor,
        tensor2d([2, 2, 2, 2, 2, 2, 2, 2], [2, 4]));

    expect(() => {
      // tslint:disable-next-line:no-unused-expression
      denseLayer.apply(symbolicInput);
    }).toThrowError(/incompatible with layer .* axis -1/);
  });
});

describe('Flatten Layer: Symbolic', () => {
  const symbolicInputs = [
    new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [14, 12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [null, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [null, 12, 10, 4], null, [], null),
  ];

  for (const symbolicInput of symbolicInputs) {
    it(`Generates correct symbolic output: no-arg constructor: ` +
           `input shape=${JSON.stringify(symbolicInput.shape)}`,
       () => {
         const flattenLayer = tfl.layers.flatten();
         const output = flattenLayer.apply(symbolicInput) as tfl.SymbolicTensor;
         const expectedShape =
             [symbolicInput.shape[0], arrayProd(symbolicInput.shape, 1)];
         expect(output.shape).toEqual(expectedShape);
         expect(output.sourceLayer).toEqual(flattenLayer);
         expect(output.inputs).toEqual([symbolicInput]);
       });

    it(`Generates correct symbolic output: empty one-arg constructor: ` +
           `input shape=${JSON.stringify(symbolicInput.shape)}`,
       () => {
         const flattenLayer = tfl.layers.flatten({});
         const output = flattenLayer.apply(symbolicInput) as tfl.SymbolicTensor;
         const expectedShape =
             [symbolicInput.shape[0], arrayProd(symbolicInput.shape, 1)];
         expect(output.shape).toEqual(expectedShape);
         expect(output.sourceLayer).toEqual(flattenLayer);
         expect(output.inputs).toEqual([symbolicInput]);
       });
  }

  it('2D tfl.SymbolicTensor leads to error', () => {
    const flattenLayer = tfl.layers.flatten();
    const x = new tfl.SymbolicTensor('float32', [null, 4], null, [], null);
    expect(() => flattenLayer.apply(x)).toThrowError();
  });

  it('3D with undetermined input size leads to error', () => {
    const flattenLayer = tfl.layers.flatten({});
    const x = new tfl.SymbolicTensor('float32', [8, 4, null], null, [], null);
    expect(() => flattenLayer.apply(x)).toThrowError(/not fully defined/);
  });
  it('Serialization round trip', () => {
    const layer = tfl.layers.flatten({dataFormat: 'channelsFirst'});
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.flatten(tsConfig);
    expect(layerPrime.getConfig().dataFormat).toEqual('channelsFirst');
  });
});

describeMathCPUAndGPU('Flatten Layer: Tensor', () => {
  it('Attempt to apply on Tensor2D leads to error', () => {
    const flattenLayer = tfl.layers.flatten();
    const x = tensor2d([[1, 3], [3, 3]], [2, 2]);
    expect(() => flattenLayer.apply(x)).toThrowError();
  });
  it('Flattens Tensor3D', () => {
    const flattenLayer = tfl.layers.flatten();
    const x =
        tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
    const expectedOutput =
        tensor2d([[10, 20, 30, 40], [-10, -20, -30, -40]], [2, 4]);
    expectTensorsClose(flattenLayer.apply(x, null) as Tensor, expectedOutput);
  });
  it('Flattens Tensor4D', () => {
    const flattenLayer = tfl.layers.flatten();
    const x = tensor4d(
        [
          [[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]],
          [[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]]
        ],
        [2, 2, 2, 2]);
    const expectedOutput = tensor2d(
        [10, 20, 30, 40, -10, -20, -30, -40, 1, 2, 3, 4, -1, -2, -3, -4],
        [2, 8]);
    expectTensorsClose(flattenLayer.apply(x, null) as Tensor, expectedOutput);
  });
  it('Flattens Tensor4D, channelFirst', () => {
    const flattenLayer = tfl.layers.flatten({dataFormat: 'channelsFirst'});
    const x = tensor4d(
        [
          [[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]],
          [[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]]
        ],
        [2, 2, 2, 2]);
    const expectedOutput = tensor2d(
        [10, -10, 20, -20, 30, -30, 40, -40, 1, -1, 2, -2, 3, -3, 4, -4],
        [2, 8]);
    expectTensorsClose(flattenLayer.apply(x, null) as Tensor, expectedOutput);
  });
});

describeMathCPUAndGPU('Activation Layer: Tensor', () => {
  const inputShape = [1];

  it('linear', () => {
    const x = mul(scalar(10), ones(inputShape));
    const layer = tfl.layers.activation({activation: 'linear'});
    const output = layer.apply(x) as Tensor;
    expectTensorsClose(output, x);
  });

  it('relu', () => {
    const x = mul(scalar(-5), ones(inputShape));
    const expectedValue = zeros(inputShape);
    const layer = tfl.layers.activation({activation: 'relu'});
    const output = layer.apply(x) as Tensor;
    expectTensorsClose(output, expectedValue);
  });

  it('sigmoid', () => {
    const val = 10;
    const x = mul(scalar(val), ones(inputShape));
    const expectedValue =
        mul(scalar(1 / (1 + Math.exp(-1 * val))), ones(inputShape));
    const layer = tfl.layers.activation({activation: 'sigmoid'});
    const output = layer.apply(x) as Tensor;
    expectTensorsClose(output, expectedValue);
  });

  it('softmax', () => {
    const x = mul(scalar(10), ones(inputShape));
    const expectedValue = ones(inputShape);
    const layer = tfl.layers.activation({activation: 'softmax'});
    const output = layer.apply(x) as Tensor;
    expectTensorsClose(output, expectedValue);
  });

  it('Serialization round trip', () => {
    const layer = tfl.layers.activation({activation: 'relu'});
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.activation(tsConfig);
    expect(layerPrime.getConfig().activation).toEqual('relu');
  });
});

describe('RepeatVector Layer: Symbolic', () => {
  it('All dimensions known.', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [3, 4], null, [], null);
    const repeatVectorLayer = tfl.layers.repeatVector({n: 2});
    const output = repeatVectorLayer.apply(symbolicInput) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([3, 2, 4]);
    expect(output.sourceLayer).toEqual(repeatVectorLayer);
    expect(output.inputs).toEqual([symbolicInput]);
  });
});

describeMathCPUAndGPU('RepeatVector Layer: Tensor', () => {
  it('With 2D tensor', () => {
    const repeatVectorLayer = tfl.layers.repeatVector({n: 3});
    const x = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const expectedOutput = tensor3d(
        [[[10, 20], [10, 20], [10, 20]], [[30, 40], [30, 40], [30, 40]]],
        [2, 3, 2]);
    expectTensorsClose(
        repeatVectorLayer.apply(x, null) as Tensor, expectedOutput);
  });
});

describe('Reshape Layer: Symbolic', () => {
  it('All dimensions known.', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
    const targetShape = [5, 8];
    const flattenLayer = tfl.layers.reshape({targetShape});
    const output = flattenLayer.apply(symbolicInput) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([12, 5, 8]);
    expect(output.sourceLayer).toEqual(flattenLayer);
    expect(output.inputs).toEqual([symbolicInput]);
  });

  it('One unknown dimension.', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
    const targetShape = [5, null];
    const flattenLayer = tfl.layers.reshape({targetShape});
    const output = flattenLayer.apply(symbolicInput) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([12, 5, 8]);
    expect(output.sourceLayer).toEqual(flattenLayer);
    expect(output.inputs).toEqual([symbolicInput]);
  });

  it('Incompatible size.', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
    const targetShape = [8, 8];
    const flattenLayer = tfl.layers.reshape({targetShape});
    expect(() => flattenLayer.apply(symbolicInput))
        .toThrowError(/Total size of new array must be unchanged/);
  });

  it('Two unknown dimensions.', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
    const targetShape: number[] = [null, null];
    const flattenLayer = tfl.layers.reshape({targetShape});
    expect(() => flattenLayer.apply(symbolicInput))
        .toThrowError(/Can only specifiy one unknown dimension/);
  });

  it('One unknown with indivisible size.', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
    const targetShape = [7, null];
    const flattenLayer = tfl.layers.reshape({targetShape});
    expect(() => flattenLayer.apply(symbolicInput))
        .toThrowError(/Total size of new array must be unchanged/);
  });

  it('Serialization round-trip', () => {
    const layer = tfl.layers.reshape({targetShape: [2, 3]});
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.reshape(tsConfig);
    expect(layerPrime.getConfig().targetShape).toEqual([2, 3]);
  });
});

describeMathCPUAndGPU('Reshape Layer: Tensor', () => {
  it('Reshape Tensor3D to Tensor3D: All dimensions known', () => {
    const reshapeLayer = tfl.layers.reshape({targetShape: [4, 1]});
    const x =
        tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
    const expectedOutput =
        tensor3d([10, 20, 30, 40, -10, -20, -30, -40], [2, 4, 1]);
    expectTensorsClose(reshapeLayer.apply(x, null) as Tensor, expectedOutput);
  });

  it('Reshape Tensor3D to Tensor2D: All dimensions known', () => {
    const reshapeLayer = tfl.layers.reshape({targetShape: [4]});
    const x =
        tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
    const expectedOutput =
        tensor2d([10, 20, 30, 40, -10, -20, -30, -40], [2, 4]);
    expectTensorsClose(reshapeLayer.apply(x, null) as Tensor, expectedOutput);
  });

  it('Reshape Tensor2D to Tensor3D: All dimensions known', () => {
    const reshapeLayer = tfl.layers.reshape({targetShape: [3, 2]});
    const x = tensor2d(
        [[10, 20, 30, 40, 50, 60], [-10, -20, -30, -40, -50, -60]], [2, 6]);
    const expectedOutput = tensor3d(
        [10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60], [2, 3, 2]);
    expectTensorsClose(reshapeLayer.apply(x, null) as Tensor, expectedOutput);
  });

  for (const unknownDim of [-1, null]) {
    it(`Reshape Tensor2D to Tensor3D: Last dimension unknown as ${unknownDim}`,
       () => {
         const reshapeLayer =
             tfl.layers.reshape({targetShape: [3, unknownDim]});
         const x = tensor2d(
             [[10, 20, 30, 40, 50, 60], [-10, -20, -30, -40, -50, -60]],
             [2, 6]);
         const expectedOutput = tensor3d(
             [10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60], [2, 3, 2]);
         expectTensorsClose(
             reshapeLayer.apply(x, null) as Tensor, expectedOutput);
       });

    it(`Reshape Tensor2D to Tensor3D: First dimension unknown as ${unknownDim}`,
       () => {
         const reshapeLayer =
             tfl.layers.reshape({targetShape: [unknownDim, 3]});
         const x = tensor2d(
             [[10, 20, 30, 40, 50, 60], [-10, -20, -30, -40, -50, -60]],
             [2, 6]);
         const expectedOutput = tensor3d(
             [10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60], [2, 2, 3]);
         expectTensorsClose(
             reshapeLayer.apply(x, null) as Tensor, expectedOutput);
       });
  }

  it('Known but incompatible dimensions', () => {
    const reshapeLayer = tfl.layers.reshape({targetShape: [3, 3]});
    const x =
        tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
    expect(() => reshapeLayer.apply(x, null))
        .toThrowError(/Total size of new array must be unchanged/);
  });

  it('Unknown and incompatible dimensions', () => {
    const reshapeLayer = tfl.layers.reshape({targetShape: [3, null]});
    const x =
        tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
    expect(() => reshapeLayer.apply(x, null))
        .toThrowError(/Total size of new array must be unchanged/);
  });

  it('More than one unknown dimension.', () => {
    const reshapeLayer = tfl.layers.reshape({targetShape: [null, null]});
    const x =
        tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
    expect(() => reshapeLayer.apply(x, null))
        .toThrowError(/Can only specifiy one unknown dimension/);
  });
});

describe('Permute Layer: Symbolic', () => {
  it('1D Trivial', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [null, 4], null, [], null);
    const dims = [1];
    const permuteLayer = tfl.layers.permute({dims});
    const output = permuteLayer.apply(symbolicInput) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([null, 4]);
    expect(output.sourceLayer).toEqual(permuteLayer);
    expect(output.inputs).toEqual([symbolicInput]);
  });

  it('2D', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [null, 4, 6], null, [], null);
    const dims = [2, 1];
    const permuteLayer = tfl.layers.permute({dims});
    const output = permuteLayer.apply(symbolicInput) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([null, 6, 4]);
    expect(output.sourceLayer).toEqual(permuteLayer);
    expect(output.inputs).toEqual([symbolicInput]);
  });

  it('3D', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [null, 4, 6, 8], null, [], null);
    const dims = [3, 1, 2];
    const permuteLayer = tfl.layers.permute({dims});
    const output = permuteLayer.apply(symbolicInput) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([null, 8, 4, 6]);
    expect(output.sourceLayer).toEqual(permuteLayer);
    expect(output.inputs).toEqual([symbolicInput]);
  });

  it('Missing dims config leads to Error', () => {
    // tslint:disable-next-line:no-any
    expect(() => tfl.layers.permute({} as any)).toThrowError(/dims.* missing/);
  });

  it('Non-Array dims config leads to Error', () => {
    // tslint:disable-next-line:no-any
    expect(() => tfl.layers.permute({dims: 1} as any))
        .toThrowError(/requires.*dims.* to be an Array/);
  });

  it('Non-consecutive dims values leads to Error', () => {
    expect(() => tfl.layers.permute({
      dims: [1, 3]
    })).toThrowError(/Invalid permutation .*dims/);
  });

  it('Repeating dims values leads to Error', () => {
    expect(() => tfl.layers.permute({
      dims: [1, 1, 3]
    })).toThrowError(/Invalid permutation .*dims/);
  });

  it('Dims values containing 0 leads to Error', () => {
    expect(() => tfl.layers.permute({
      dims: [0, 1, 2]
    })).toThrowError(/Invalid permutation .*dims/);
  });

  it('Serialization round-trip', () => {
    const layer = tfl.layers.permute({dims: [1, 3, 2]});
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.permute(tsConfig);
    expect(layerPrime.getConfig().dims).toEqual([1, 3, 2]);
  });
});

describe('Masking Layer: Symbolic', () => {
  it('computeOutputShape', () => {
    const layer = tfl.layers.masking();
    const inputShape = [null, 4, 6];
    const outputShape = layer.computeOutputShape(inputShape);
    expect(outputShape).toEqual([null, 4, 6]);
  });

  it('Serialization round-trip', () => {
    const layer = tfl.layers.masking({maskValue: -3});
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.masking(tsConfig);
    expect(layerPrime.getConfig().maskValue).toEqual(-3);
  });
});

describeMathCPUAndGPU('Permute Layer: Tensor', () => {
  it('2D', () => {
    const permuteLayer = tfl.layers.permute({dims: [2, 1]});
    const x =
        tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
    const expectedOutput =
        tensor3d([[[10, 30], [20, 40]], [[-10, -30], [-20, -40]]], [2, 2, 2]);
    expectTensorsClose(permuteLayer.apply(x) as Tensor, expectedOutput);
  });
});

describeMathCPUAndWebGL2('Masking Layer: Tensor', () => {
  // Reference Python code:
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Masking(input_shape=[3, 2]))
  // model.add(tf.keras.layers.SimpleRNN(
  //   units=1,
  //   recurrent_initializer='ones',
  //   kernel_initializer='ones'))
  //
  // xs = np.array([[[1, 1], [1, 0], [0, 0]]], dtype=np.float32)
  // print(xs.shape)
  // ys = model.predict(xs)
  //
  // print(ys)
  // ```
  it('3D, default maskValue', () => {
    const model = tfl.sequential();
    model.add(tfl.layers.masking({inputShape: [3, 2]}));
    model.add(tfl.layers.simpleRNN(
        {units: 1, recurrentInitializer: 'ones', kernelInitializer: 'ones'}));

    const xs = tensor3d([[[1, 1], [1, 0], [0, 0]]]);
    const ys = model.predict(xs) as Tensor;
    expectTensorsClose(ys, tensor2d([[0.961396]]));
  });

  // Reference Python code:
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Masking(mask_value=-1, input_shape=[3, 2]))
  // model.add(tf.keras.layers.SimpleRNN(
  //   units=1,
  //   recurrent_initializer='ones',
  //   kernel_initializer='ones'))
  //
  // xs = np.array([[[1, 1], [1, -1], [-1, -1]]], dtype=np.float32)
  // print(xs.shape)
  // ys = model.predict(xs)
  //
  // print(ys)
  // ```
  it('3D, custom maskValue', () => {
    const model = tfl.sequential();
    model.add(tfl.layers.masking({maskValue: -1, inputShape: [3, 2]}));
    model.add(tfl.layers.simpleRNN(
        {units: 1, recurrentInitializer: 'ones', kernelInitializer: 'ones'}));

    const xs = tensor3d([[[1, 1], [1, -1], [-1, -1]]]);
    const ys = model.predict(xs) as Tensor;
    expectTensorsClose(ys, tensor2d([[0.746068]]));
  });
});
