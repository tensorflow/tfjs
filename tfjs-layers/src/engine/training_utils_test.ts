/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {memory, tensor1d, tensor2d} from '@tensorflow/tfjs-core';

import {describeMathCPU, expectTensorsClose} from '../utils/test_utils';

import {ClassWeight, ClassWeightMap, standardizeClassWeights, standardizeWeights} from './training_utils';

describeMathCPU('standardizeWeights', () => {
  it('classWeights with 1D class-index target', async () => {
    const y = tensor1d([0, 1, 2, 1, 0]);
    const classWeight: ClassWeight = {0: 10, 1: 1, 2: 0.1};
    const numTensors0 = memory().numTensors;
    const classSampleWeight = await standardizeWeights(y, null, classWeight);
    // Assert no memory leak. The extra tensor is `classSampleWeight` itself.
    expect(memory().numTensors).toEqual(numTensors0 + 1);
    expectTensorsClose(classSampleWeight, tensor1d([10, 1, 0.1, 1, 10]));
    expect(y.isDisposed).toEqual(false);
  });

  it('classWeights with 2D class-index target', async () => {
    const y = tensor2d([[3], [2], [0]]);
    const classWeight: ClassWeight = {0: 10, 1: 1, 2: 0.1, 3: 0.01};
    const numTensors0 = memory().numTensors;
    const classSampleWeight = await standardizeWeights(y, null, classWeight);
    // Assert no memory leak. The extra tensor is `classSampleWeight` itself.
    expect(memory().numTensors).toEqual(numTensors0 + 1);
    expectTensorsClose(classSampleWeight, tensor1d([0.01, 0.1, 10]));
    expect(y.isDisposed).toEqual(false);
  });

  it('classWeights with 2D one-hot target', async () => {
    const y = tensor2d([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]]);
    const classWeight: ClassWeight = {0: 10, 1: 1, 2: 0.1, 3: 0.01};
    const numTensors0 = memory().numTensors;
    const classSampleWeight = await standardizeWeights(y, null, classWeight);
    // Assert no memory leak. The extra tensor is `classSampleWeight` itself.
    expect(memory().numTensors).toEqual(numTensors0 + 1);
    expectTensorsClose(classSampleWeight, tensor1d([0.01, 0.1, 10]));
    expect(y.isDisposed).toEqual(false);
  });

  it('classWeights with 1D class-index target: Missing class', async () => {
    const y = tensor1d([0, 1, 2, 3, 2, 1, 0]);
    const classWeight: ClassWeight = {0: 10, 1: 1, 2: 0.1};

    let caughtError: Error;
    try {
      await standardizeWeights(y, null, classWeight);
    } catch (error) {
      caughtError = error;
    }
    expect(caughtError.message)
        .toMatch(/classWeight must contain all classes.* class 3 .*/);
  });

  it('classWeights with 2D class-index target: Missing class', async () => {
    const y = tensor2d([[3], [2], [0], [4]]);
    const classWeight: ClassWeight = {0: 10, 1: 1, 2: 0.1, 3: 0.01};

    let caughtError: Error;
    try {
      await standardizeWeights(y, null, classWeight);
    } catch (error) {
      caughtError = error;
    }
    expect(caughtError.message)
        .toMatch(/classWeight must contain all classes.* class 4 .*/);
  });

  it('classWeights with 2D one-hot target: missing weight', async () => {
    const y = tensor2d([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]]);
    const classWeight: ClassWeight = {0: 10, 1: 1, 3: 0.01};

    let caughtError: Error;
    try {
      await standardizeWeights(y, null, classWeight);
    } catch (error) {
      caughtError = error;
    }
    expect(caughtError.message)
        .toMatch(/classWeight must contain all classes.* class 2 .*/);
  });
});

describe('standardizeClassWeights', () => {
  it('One output, ClassWeight singleton', () => {
    const outputNames = ['output1'];
    const classWeight: ClassWeight = {0: 1, 1: 2};
    const output = standardizeClassWeights(classWeight, outputNames);
    expect(output).toEqual([{0: 1, 1: 2}]);
  });

  it('One output, ClassWeight array', () => {
    const outputNames = ['output1'];
    const classWeight: ClassWeight[] = [{0: 1, 1: 2}];
    const output = standardizeClassWeights(classWeight, outputNames);
    expect(output).toEqual([{0: 1, 1: 2}]);
  });

  it('One output, ClassWeight dict', () => {
    const outputNames = ['output1'];
    const classWeight: ClassWeightMap = {'output1': {0: 1, 1: 2}};
    const output = standardizeClassWeights(classWeight, outputNames);
    expect(output).toEqual([{0: 1, 1: 2}]);
  });

  it('Two outputs, ClassWeight array', () => {
    const outputNames = ['output1', 'output2'];
    const classWeight: ClassWeight[] = [{0: 1, 1: 2}, {0: 10, 1: 20}];
    const output = standardizeClassWeights(classWeight, outputNames);
    expect(output).toEqual([{0: 1, 1: 2}, {0: 10, 1: 20}]);
  });

  it('Two outputs, ClassWeight dict', () => {
    const outputNames = ['output1', 'output2'];
    const classWeight:
        ClassWeightMap = {'output2': {0: 10, 1: 20}, 'output1': {0: 1, 1: 2}};
    const output = standardizeClassWeights(classWeight, outputNames);
    expect(output).toEqual([{0: 1, 1: 2}, {0: 10, 1: 20}]);
  });

  it('Two outputs, ClassWeight singleton leads to Error', () => {
    const outputNames = ['output1', 'output2'];
    const classWeight: ClassWeight = {0: 10, 1: 20};
    expect(() => standardizeClassWeights(classWeight, outputNames))
        .toThrowError(/.*has multiple \(2\) outputs.*/);
  });

  it('Three outputs, ClassWeight array missing element', () => {
    const outputNames = ['output1', 'output2', 'output3'];
    const classWeight: ClassWeight[] = [{0: 1, 1: 2}, {0: 10, 1: 20}];
    expect(() => standardizeClassWeights(classWeight, outputNames))
        .toThrowError(
            /.*classWeight is an array of 2 element.* model has 3 outputs/);
  });

  it('Three outputs, ClassWeight dict missing element is okay', () => {
    const outputNames = ['output1', 'output2', 'output3'];
    const classWeight:
        ClassWeightMap = {'output1': {0: 1, 1: 2}, 'output3': {0: 10, 1: 20}};
    const output = standardizeClassWeights(classWeight, outputNames);
    expect(output).toEqual([{0: 1, 1: 2}, null, {0: 10, 1: 20}]);
  });
});
