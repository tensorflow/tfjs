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

import * as tfc from '@tensorflow/tfjs-core';
import {scalar, Tensor, tensor1d, tensor2d} from '@tensorflow/tfjs-core';

import {epsilon} from './backend/common';
import * as losses from './losses';
import {describeMathCPUAndGPU, expectTensorsClose} from './utils/test_utils';

describeMathCPUAndGPU('meanSquaredError', () => {
  it('1D', () => {
    const yTrue = tfc.zeros([3]);
    const yPred = tensor1d([1, 2, 3]);
    const expectedVal = scalar((1 * 1 + 2 * 2 + 3 * 3) / 3);
    const result = losses.meanSquaredError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });

  it('2D', () => {
    const yTrue = tfc.zeros([2, 2]);
    const yPred = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const expectedVal = tensor1d([(1 * 1 + 2 * 2) / 2, (3 * 3 + 4 * 4) / 2]);
    const result = losses.meanSquaredError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('meanAbsoluteError', () => {
  it('1D', () => {
    const yTrue = tfc.zeros([3]);
    const yPred = tensor1d([-1, -2, -3]);
    const expectedVal = scalar((1 + 2 + 3) / 3);
    const result = losses.meanAbsoluteError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });

  it('2D', () => {
    const yTrue = tfc.zeros([2, 2]);
    const yPred = tensor2d([[-1, -2], [-3, -4]], [2, 2]);
    const expectedVal = tensor1d([(1 + 2) / 2, (3 + 4) / 2]);
    const result = losses.meanAbsoluteError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('meanAbsolutePercentageError', () => {
  it('1D', () => {
    const yTrue = tensor1d([-1, -2, -3]);
    const yPred = tfc.zeros([3]);
    const expectedVal = scalar((1 + 2 + 3) / (1 + 2 + 3) * 100);
    const result = losses.meanAbsolutePercentageError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });

  it('2D', () => {
    const yTrue = tensor2d([[-1, -2], [-3, -4]], [2, 2]);
    const yPred = tfc.zeros([2, 2]);
    const expectedVal =
        tensor1d([(1 + 2) / (1 + 2) * 100, (3 + 4) / (3 + 4) * 100]);
    const result = losses.meanAbsolutePercentageError(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('meanSquaredLogarithmicError', () => {
  function meanSquaredLogErrorFor1DArray(x: number[], y: number[]): number {
    const calcLog = (val: number) => Math.log(Math.max(val, epsilon()) + 1);
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
    const yTrue = tfc.zeros([2, 2]);
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
    const yTrue = tfc.zeros([2, 2]);
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


describeMathCPUAndGPU('categoricalCrossentropy ', () => {
  it('from logits', () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
    const expected = tensor1d([
      -1 *
          (Math.log(Math.exp(1) / (Math.exp(1) + Math.exp(2))) * 0.25 +
           Math.log(Math.exp(2) / (Math.exp(1) + Math.exp(2))) * 0.75),
      -1 *
          (Math.log(Math.exp(3) / (Math.exp(3) + Math.exp(4))) * 0.1 +
           Math.log(Math.exp(4) / (Math.exp(3) + Math.exp(4))) * 0.9)
    ]);
    const result = losses.categoricalCrossentropy(target, x, true);
    expectTensorsClose(result, expected);
  });

  it('from softmax', () => {
    const x = tensor2d([[0.3, 0.7], [0.4, 0.6]], [2, 2]);
    const target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
    const expected = tensor1d([
      -1 * (Math.log(0.3) * 0.25 + Math.log(0.7) * 0.75),
      -1 * (Math.log(0.4) * 0.1 + Math.log(0.6) * 0.9)
    ]);
    const result = losses.categoricalCrossentropy(target, x, false);
    expectTensorsClose(result, expected);
  });
});

describeMathCPUAndGPU('sparseCategoricalCrossentropy ', () => {
  // Reference Python TensorFlow code:
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // with tf.Session() as sess:
  //   x = tf.placeholder(tf.float32, [None, 3])
  //   target = tf.placeholder(tf.float32, [None])
  //   crossentropy = tf.keras.backend.sparse_categorical_crossentropy(
  //       target, x)
  //   out = sess.run(
  //       crossentropy,
  //       feed_dict={
  //           x: np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]],
  //                       dtype=np.float32),
  //           target: np.array([0, 2], dtype=np.float32)
  //           })
  //   print(out)
  // ```
  it('sparseCategoricalCrossentropy', () => {
    const x = tensor2d([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]], [2, 3]);
    const target = tensor1d([0, 2]);
    const result = losses.sparseCategoricalCrossentropy(target, x);
    expectTensorsClose(result, tensor1d([2.3025851, 0.6931472]));
  });
});

describeMathCPUAndGPU('sigmoidCrossEntropyWithLogits', () => {
  it('outputs sigmoid cross-entropy', () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
    const targetComplement = tfc.add(scalar(1), tfc.neg(target));
    const sigmoidX = tfc.sigmoid(x);
    const sigmoidXComplement = tfc.add(scalar(1), tfc.neg(sigmoidX));
    const expected = tfc.add(
        tfc.mul(target, tfc.neg(tfc.log(sigmoidX))),
        tfc.mul(targetComplement, tfc.neg(tfc.log(sigmoidXComplement))));
    const result = losses.sigmoidCrossEntropyWithLogits(target, x);
    expectTensorsClose(result, expected);
  });

  // Python TensorFlow reference code:
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // logits = np.array([[-10, -10, -10],
  //                    [-5, -5, -5],
  //                    [0, 0, 0],
  //                    [0.5, 0.5, 0.5],
  //                    [2, 2, 2]], dtype=np.float32)
  // labels = np.array([[0, 0.5, 1],
  //                    [0, 0.5, 1],
  //                    [0, 0.5, 1],
  //                    [0, 0.5, 1],
  //                    [0, 0.5, 1]], dtype=np.float32)
  //
  // print(tf.nn.sigmoid_cross_entropy_with_logits(
  //     logits=logits, labels=labels))
  // ```
  it('Comparison with TensorFlow references values', () => {
    const logits = tensor2d(
        [[-10, -10, -10],
         [-5, -5, -5],
         [0, 0, 0],
         [0.5, 0.5, 0.5],
         [2, 2, 2]]);
    const labels = tensor2d(
        [[0, 0.5, 1],
         [0, 0.5, 1],
         [0, 0.5, 1],
         [0, 0.5, 1],
         [0, 0.5, 1]]);
    const outputs = losses.sigmoidCrossEntropyWithLogits(labels, logits);
    expectTensorsClose(outputs, tensor2d(
        [[4.5398901e-05, 5.0000453e+00, 1.0000046e+01],
         [6.7153485e-03, 2.5067153e+00, 5.0067153e+00],
         [6.9314718e-01, 6.9314718e-01, 6.9314718e-01],
         [9.7407699e-01, 7.2407699e-01, 4.7407699e-01],
         [2.1269281e+00, 1.1269280e+00, 1.2692800e-01]]));
  });
});


describeMathCPUAndGPU('categoricalCrossentropy', () => {
  it('2D', () => {
    const yTrue = tensor2d([[1, 0], [0, 1]], [2, 2]);
    const yPred = yTrue;
    const expectedVal = tfc.zeros([2]);
    const result = losses.categoricalCrossentropy(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('sparseCategoricalCrossentropy', () => {
  it('2D', () => {
    const yTrue = tensor1d([0, 1]);
    const yPred = tensor2d([[1, 0], [0, 1]], [2, 2]);
    const expectedVal = tfc.zeros([2]);
    const result = losses.sparseCategoricalCrossentropy(yTrue, yPred);
    expectTensorsClose(result, expectedVal);
  });
});

describeMathCPUAndGPU('binaryCrossentropy', () => {
  function _binaryCrossentropy(target: Tensor, output: Tensor): Tensor {
    const targetComplement = tfc.add(scalar(1), tfc.neg(target));
    const outputComplement = tfc.add(scalar(1), tfc.neg(output));
    return tfc.mean(
        tfc.neg(tfc.add(
            tfc.mul(target, tfc.log(output)),
            tfc.mul(targetComplement, tfc.log(outputComplement)))),
        -1);
  }

  it('from sigmoid', () => {
    const x = tensor2d([[0.3, 0.7], [0.4, 0.6]], [2, 2]);
    const target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
    const expected = _binaryCrossentropy(target, x);
    const result = losses.binaryCrossentropy(target, x);
    expectTensorsClose(result, expected);
  });
});

describeMathCPUAndGPU('kullbackLeiblerDivergence', () => {
  function klElement(actual: number, predicted: number): number {
    actual = Math.max(actual, epsilon());
    predicted = Math.max(predicted, epsilon());
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
    return predicted - actual * Math.log(predicted + epsilon());
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

  it(`get custom loss works`, () => {
    const customLoss = (x: Tensor, y: Tensor) => scalar(42.0);
    expect(losses.get(customLoss)).toEqual(customLoss);
  });
});

describeMathCPUAndGPU('l2Normalize', () => {
  it('normalizes with no axis defined.', () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const norm = Math.sqrt(1 * 1 + 2 * 2 + 3 * 3 + 4 * 4);
    const expected =
        tensor2d([[1 / norm, 2 / norm], [3 / norm, 4 / norm]], [2, 2]);
    const result = losses.l2Normalize(x);
    expectTensorsClose(result, expected);
  });

  it('normalizes along axis = -1.', () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const firstNorm = Math.sqrt(1 * 1 + 2 * 2);
    const secondNorm = Math.sqrt(3 * 3 + 4 * 4);
    const expected = tensor2d(
        [[1 / firstNorm, 2 / firstNorm], [3 / secondNorm, 4 / secondNorm]],
        [2, 2]);
    const result = losses.l2Normalize(x, -1);
    expectTensorsClose(result, expected);
  });

  it('normalizes with zeros.', () => {
    const x = tfc.zeros([2, 2]);
    const result = losses.l2Normalize(x);
    expectTensorsClose(result, x);
  });
});
