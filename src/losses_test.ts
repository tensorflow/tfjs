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
 * Unit tests for activations.ts.
 */

import {scalar, tensor1d, tensor2d, zeros} from '@tensorflow/tfjs-core';

import * as K from './backend/deeplearnjs_backend';
import * as losses from './losses';
import {describeMathCPUAndGPU, expectTensorsClose} from './utils/test_utils';

describeMathCPUAndGPU('meanSquaredError', () => {
  it('1D', () => {
    const yTrue = zeros([3]);
    const yPred = tensor1d([1, 2, 3]);
    const expectedVal = scalar((1 * 1 + 2 * 2 + 3 * 3) / 3);
    const result = losses.meanSquaredError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });

  it('2D', () => {
    const yTrue = zeros([2, 2]);
    const yPred = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const expectedVal = tensor1d([(1 * 1 + 2 * 2) / 2, (3 * 3 + 4 * 4) / 2]);
    const result = losses.meanSquaredError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('meanAbsoluteError', () => {
  it('1D', () => {
    const yTrue = zeros([3]);
    const yPred = tensor1d([-1, -2, -3]);
    const expectedVal = scalar((1 + 2 + 3) / 3);
    const result = losses.meanAbsoluteError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });

  it('2D', () => {
    const yTrue = zeros([2, 2]);
    const yPred = tensor2d([[-1, -2], [-3, -4]], [2, 2]);
    const expectedVal = tensor1d([(1 + 2) / 2, (3 + 4) / 2]);
    const result = losses.meanAbsoluteError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('meanAbsolutePercentageError', () => {
  it('1D', () => {
    const yTrue = tensor1d([-1, -2, -3]);
    const yPred = zeros([3]);
    const expectedVal = scalar((1 + 2 + 3) / (1 + 2 + 3) * 100);
    const result = losses.meanAbsolutePercentageError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });

  it('2D', () => {
    const yTrue = tensor2d([[-1, -2], [-3, -4]], [2, 2]);
    const yPred = zeros([2, 2]);
    const expectedVal =
        tensor1d([(1 + 2) / (1 + 2) * 100, (3 + 4) / (3 + 4) * 100]);
    const result = losses.meanAbsolutePercentageError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('meanSquaredLogarithmicError', () => {
  function meanSquaredLogErrorFor1DArray(x: number[], y: number[]): number {
    const calcLog = (val: number) => Math.log(Math.max(val, K.epsilon()) + 1);
    const logX = x.map(calcLog);
    const logY = y.map(calcLog);
    let acc = 0.0;
    for (let i = 0; i < x.length; i++) {
      const diff = logX[i] - logY[i];
      acc += diff * diff;
    }
    return acc / x.length;
  }

  it('2D', () => {
    const yTrue = zeros([2, 2]);
    const yPred = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const expectedVal = tensor1d([
      meanSquaredLogErrorFor1DArray([1, 2], [0, 0]),
      meanSquaredLogErrorFor1DArray([3, 4], [0, 0])
    ]);
    const result = losses.meanSquaredLogarithmicError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('squaredHinge', () => {
  it('2D', () => {
    const yTrue = tensor2d([[-1, 2], [-3, 2]], [2, 2]);
    const yPred = tensor2d([[-3, 5], [3, -2]], [2, 2]);
    // First row has correct predictions, so loss is 0.
    const secondRow = [1 - (-3 * 3), 1 - (2 * -2)].map(x => x * x);
    const expectedVal = tensor1d([0, (secondRow[0] + secondRow[1]) / 2]);
    const result = losses.squaredHinge(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('hinge', () => {
  it('2D', () => {
    const yTrue = tensor2d([[-1, 2], [-3, 2]], [2, 2]);
    const yPred = tensor2d([[-3, 5], [3, -2]], [2, 2]);
    // First row has correct predictions, so loss is 0.
    const secondRow = [1 - (-3 * 3), 1 - (2 * -2)];
    const expectedVal = tensor1d([0, (secondRow[0] + secondRow[1]) / 2]);
    const result = losses.hinge(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('categoricalHinge', () => {
  it('2D', () => {
    const yTrue = tensor2d([[0, 1, 0], [1, 0, 0]], [2, 3]);
    const yPred = tensor2d([[0, 2, 0], [1, 3, 2]], [2, 3]);
    // First row has correct predictions, so loss is 0.
    const secondRowPos = 1 * 1;
    const secondRowNeg = Math.max(1 * 3, 1 * 2);
    const secondRowVal = secondRowNeg - secondRowPos + 1;
    const expectedVal = tensor1d([0, secondRowVal]);
    const result = losses.categoricalHinge(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('logcosh', () => {
  function _logcosh(x: number): number {
    return x + Math.log(Math.exp(-2 * x) + 1) - Math.log(2);
  }
  it('2D', () => {
    const yTrue = zeros([2, 2]);
    const yPred = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const firstRow = [1, 2].map(_logcosh);
    const secondRow = [3, 4].map(_logcosh);
    const firstVal = (firstRow[0] + firstRow[1]) / 2;
    const secondVal = (secondRow[0] + secondRow[1]) / 2;
    const expectedVal = tensor1d([firstVal, secondVal]);
    const result = losses.logcosh(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('categoricalCrossentropy', () => {
  it('2D', () => {
    const yTrue = tensor2d([[1, 0], [0, 1]], [2, 2]);
    const yPred = yTrue;
    const expectedVal = zeros([2]);
    const result = losses.categoricalCrossentropy(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('sparseCategoricalCrossentropy', () => {
  it('2D', () => {
    const yTrue = tensor1d([0, 1]);
    const yPred = tensor2d([[1, 0], [0, 1]], [2, 2]);
    const expectedVal = zeros([2]);
    const result = losses.sparseCategoricalCrossentropy(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('binaryCrossentropy', () => {
  it('2D', () => {
    const yTrue = tensor2d([[1, 0], [1, 0]], [2, 2]);
    const yPred = tensor2d([[1, 2], [20, 10]], [2, 2]);
    const crossEntropy = K.binaryCrossentropy(yTrue, yPred).dataSync();
    const expectedVal = tensor1d([
      (crossEntropy[0] + crossEntropy[1]) / 2,
      (crossEntropy[2] + crossEntropy[3]) / 2
    ]);
    const result = losses.binaryCrossentropy(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('kullbackLeiblerDivergence', () => {
  function klElement(actual: number, predicted: number): number {
    actual = Math.max(actual, K.epsilon());
    predicted = Math.max(predicted, K.epsilon());
    return actual * Math.log(actual / predicted);
  }

  it('2D', () => {
    const yTrue = tensor2d([[1, 0], [1, 0]], [2, 2]);
    const yPred = tensor2d([[0.25, 0.75], [0.9, 0.1]], [2, 2]);
    const expectedVal = tensor1d([
      klElement(1, 0.25) + klElement(0, 0.75),
      klElement(1, 0.9) + klElement(0, 0.1),
    ]);
    const result = losses.kullbackLeiblerDivergence(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('poisson', () => {
  function poissonElement(actual: number, predicted: number): number {
    return predicted - actual * Math.log(predicted + K.epsilon());
  }

  it('2D', () => {
    const yTrue = tensor2d([[1, 0], [1, 0]], [2, 2]);
    const yPred = tensor2d([[0.25, 0.75], [0.9, 0.1]], [2, 2]);
    const expectedVal = tensor1d([
      (poissonElement(1, 0.25) + poissonElement(0, 0.75)) / 2,
      (poissonElement(1, 0.9) + poissonElement(0, 0.1)) / 2,
    ]);
    const result = losses.poisson(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('cosineProximity', () => {
  it('2D', () => {
    const z = Math.sqrt(2) / 2;
    const yTrue = tensor2d([[1, 0], [1, 0]], [2, 2]);
    const yPred = tensor2d([[z, z], [0, 1]], [2, 2]);
    const expectedVal = tensor1d([-1 * z, 0]);
    const result = losses.cosineProximity(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});


describe('losses get', () => {
  for (const lossName
           of ['meanSquaredError', 'meanAbsoluteError',
               'meanAbsolutePercentageError', 'meanSquaredLogarithmicError',
               'squaredHinge', 'hinge', 'categoricalHinge', 'logcosh',
               'categoricalCrossentropy', 'sparseCategoricalCrossentropy',
               'binaryCrossentropy', 'kullbackLeiblerDivergence', 'poisson',
               'cosineProximity']) {
    it(`can get ${lossName}`, () => {
      losses.get(lossName);
    });
  }
});
