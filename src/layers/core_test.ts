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

// tslint:disable:max-line-length
import {ones, scalar, Tensor, tensor2d, tensor3d, tensor4d, zeros} from '@tensorflow/tfjs-core';

import {ActivationIdentifier} from '../activations';
import * as K from '../backend/tfjs_backend';
import * as tfl from '../index';
import {InitializerIdentifier} from '../initializers';
import {pyListRepeat} from '../utils/generic_utils';
import {arrayProd} from '../utils/math_utils';
import {convertPythonicToTs, convertTsToPythonic} from '../utils/serialization_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {Activation, RepeatVector, Reshape} from './core';
// tslint:enable:max-line-length

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
  it('tensor', () => {
    const inputShape = [2, 3, 4];
    const trainingValues = [false, true];
    const dropoutRates = [0, 0.5];
    const noiseShapes = [null, inputShape];
    // TODO(cais): test non-default noiseShapes once they are supported.

    for (const training of trainingValues) {
      for (const rate of dropoutRates) {
        for (const noiseShape of noiseShapes) {
          const testTitle = `training=${training}, dropoutRate=${rate}, ` +
              `noiseShape=${JSON.stringify(noiseShape)}`;
          it(testTitle, () => {
            const x = ones(inputShape);
            const dropoutLayer = tfl.layers.dropout({rate, noiseShape});
            const y = dropoutLayer.apply(x, {training}) as Tensor;
            expect(x.dtype).toEqual(y.dtype);
            expect(x.shape).toEqual(y.shape);
            const xValue = x.dataSync();
            const yValue = y.dataSync();
            let nKept = 0;
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
});

describeMathCPU('Dense Layer: Symbolic', () => {
  const units = 3;
  const activations = [null, 'linear', 'relu', 'softmax'];
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
      // tslint:disable-next-line:no-unused-expression
      tfl.layers.dense({units: 4, activation: 'invalid_ativation!'});
    }).toThrowError(/Unknown activation/);
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
               if (K.ndim(input) === 2) {
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
});

describeMathCPUAndGPU('Activation Layer: Tensor', () => {
  const inputShape = [1];

  it('linear', () => {
    const x = K.scalarTimesArray(scalar(10), ones(inputShape));
    const layer = new Activation({activation: 'linear'});
    const output = layer.apply(x) as Tensor;
    expectTensorsClose(output, x);
  });

  it('relu', () => {
    const x = K.scalarTimesArray(scalar(-5), ones(inputShape));
    const expectedValue = zeros(inputShape);
    const layer = new Activation({activation: 'relu'});
    const output = layer.apply(x) as Tensor;
    expectTensorsClose(output, expectedValue);
  });

  it('sigmoid', () => {
    const val = 10;
    const x = K.scalarTimesArray(scalar(val), ones(inputShape));
    const expectedValue = K.scalarTimesArray(
        scalar(1 / (1 + Math.exp(-1 * val))), ones(inputShape));
    const layer = new Activation({activation: 'sigmoid'});
    const output = layer.apply(x) as Tensor;
    expectTensorsClose(output, expectedValue);
  });

  it('softmax', () => {
    const x = K.scalarTimesArray(scalar(10), ones(inputShape));
    const expectedValue = ones(inputShape);
    const layer = new Activation({activation: 'softmax'});
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
    const repeatVectorLayer = new RepeatVector({n: 2});
    const output = repeatVectorLayer.apply(symbolicInput) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([3, 2, 4]);
    expect(output.sourceLayer).toEqual(repeatVectorLayer);
    expect(output.inputs).toEqual([symbolicInput]);
  });
});

describeMathCPUAndGPU('RepeatVector Layer: Tensor', () => {
  it('With 2D tensor', () => {
    const repeatVectorLayer = new RepeatVector({n: 3});
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
    const flattenLayer = new Reshape({targetShape});
    const output = flattenLayer.apply(symbolicInput) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([12, 5, 8]);
    expect(output.sourceLayer).toEqual(flattenLayer);
    expect(output.inputs).toEqual([symbolicInput]);
  });

  it('One unknown dimension.', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
    const targetShape = [5, null];
    const flattenLayer = new Reshape({targetShape});
    const output = flattenLayer.apply(symbolicInput) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([12, 5, 8]);
    expect(output.sourceLayer).toEqual(flattenLayer);
    expect(output.inputs).toEqual([symbolicInput]);
  });

  it('Incompatible size.', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
    const targetShape = [8, 8];
    const flattenLayer = new Reshape({targetShape});
    expect(() => flattenLayer.apply(symbolicInput))
        .toThrowError(/Total size of new array must be unchanged/);
  });

  it('Two unknown dimensions.', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
    const targetShape: number[] = [null, null];
    const flattenLayer = new Reshape({targetShape});
    expect(() => flattenLayer.apply(symbolicInput))
        .toThrowError(/Can only specifiy one unknown dimension/);
  });

  it('One unknown with indivisible size.', () => {
    const symbolicInput =
        new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
    const targetShape = [7, null];
    const flattenLayer = new Reshape({targetShape});
    expect(() => flattenLayer.apply(symbolicInput))
        .toThrowError(/Total size of new array must be unchanged/);
  });
});

describeMathCPUAndGPU('Reshape Layer: Tensor', () => {
  it('Reshape Tensor3D to Tensor3D: All dimensions known', () => {
    const reshapeLayer = new Reshape({targetShape: [4, 1]});
    const x =
        tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
    const expectedOutput =
        tensor3d([10, 20, 30, 40, -10, -20, -30, -40], [2, 4, 1]);
    expectTensorsClose(reshapeLayer.apply(x, null) as Tensor, expectedOutput);
  });

  it('Reshape Tensor3D to Tensor2D: All dimensions known', () => {
    const reshapeLayer = new Reshape({targetShape: [4]});
    const x =
        tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
    const expectedOutput =
        tensor2d([10, 20, 30, 40, -10, -20, -30, -40], [2, 4]);
    expectTensorsClose(reshapeLayer.apply(x, null) as Tensor, expectedOutput);
  });

  it('Reshape Tensor2D to Tensor3D: All dimensions known', () => {
    const reshapeLayer = new Reshape({targetShape: [3, 2]});
    const x = tensor2d(
        [[10, 20, 30, 40, 50, 60], [-10, -20, -30, -40, -50, -60]], [2, 6]);
    const expectedOutput = tensor3d(
        [10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60], [2, 3, 2]);
    expectTensorsClose(reshapeLayer.apply(x, null) as Tensor, expectedOutput);
  });

  for (const unknownDim of [-1, null]) {
    it(`Reshape Tensor2D to Tensor3D: Last dimension unknown as ${unknownDim}`,
       () => {
         const reshapeLayer = new Reshape({targetShape: [3, unknownDim]});
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
         const reshapeLayer = new Reshape({targetShape: [unknownDim, 3]});
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
    const reshapeLayer = new Reshape({targetShape: [3, 3]});
    const x =
        tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
    expect(() => reshapeLayer.apply(x, null))
        .toThrowError(/Total size of new array must be unchanged/);
  });

  it('Unknown and incompatible dimensions', () => {
    const reshapeLayer = new Reshape({targetShape: [3, null]});
    const x =
        tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
    expect(() => reshapeLayer.apply(x, null))
        .toThrowError(/Total size of new array must be unchanged/);
  });

  it('More than one unknown dimension.', () => {
    const reshapeLayer = new Reshape({targetShape: [null, null]});
    const x =
        tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
    expect(() => reshapeLayer.apply(x, null))
        .toThrowError(/Can only specifiy one unknown dimension/);
  });
});
