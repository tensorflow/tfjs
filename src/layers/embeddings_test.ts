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
 * Unit tests for embedding.ts.
 */

// tslint:disable:max-line-length
import {Tensor, tensor3d} from 'deeplearn';
import {expectArraysClose} from 'deeplearn/dist/test_util';

import {DType} from '../types';
import {SymbolicTensor} from '../types';
import {describeMathCPU} from '../utils/test_utils';

import {Embedding} from './embeddings';

// tslint:enable:max-line-length

describeMathCPU('Embedding Layers: Symbolic 1D, 2D & 3D', () => {
  const inputShapes = [[1], [5], [1, 20], [2, 3, 4]];
  const outputDims = [1, 7, 47];
  const batchDim = 17;
  const inputDim = 100;
  for (const inputShape of inputShapes) {
    for (const outputDim of outputDims) {
      const testTitle = `inputShape=${inputShape}, outputDim=${outputDim}`;
      it(testTitle, () => {
        const embeddingLayer = new Embedding({inputDim, outputDim});
        const fullInputShape = [batchDim].concat(inputShape);
        const symbolicInput =
            new SymbolicTensor(DType.float32, fullInputShape, null, [], null);
        const output = embeddingLayer.apply(symbolicInput) as SymbolicTensor;
        const expectedShape = [batchDim].concat(inputShape).concat([outputDim]);
        expect(output.shape).toEqual(expectedShape);
      });
    }
  }
});

describeMathCPU('Embedding Layers: With explicit inputLength', () => {
  const inputShape = [null, 4, 5];
  const outputDim = 7;
  const inputLengths = [[null, 4, 5], [null, null, 5], [null, null, null]];
  const batchDim = 17;
  for (const inputLength of inputLengths) {
    const testTitle = `inputLength=${inputLength}`;
    it(testTitle, () => {
      const inputDim = 100;
      const embeddingLayer = new Embedding({inputDim, outputDim, inputLength});
      const fullInputShape = [batchDim].concat(inputShape);
      const symbolicInput =
          new SymbolicTensor(DType.float32, fullInputShape, null, [], null);
      const output = embeddingLayer.apply(symbolicInput) as SymbolicTensor;
      const expectedShape = [batchDim].concat(inputShape).concat([outputDim]);
      expect(output.shape).toEqual(expectedShape);
      expect(output.dtype).toEqual(symbolicInput.dtype);
    });
  }
});


describeMathCPU('Embedding Layers: Tensor', () => {
  it('check value equality', () => {
    const x = tensor3d([0, 5, 1, 1, 1, 1, 1, 1], [1, 2, 4]);
    const embeddingLayer = new Embedding({
      inputDim: 6,
      outputDim: 3,
      embeddingsInitializer: 'RandomUniform',
    });
    const y = embeddingLayer.apply(x) as Tensor;
    const yExpectedShape = [1, 2, 4, 3];
    expect(y.shape).toEqual(yExpectedShape);
    const weights = embeddingLayer.getWeights()[0];
    expect(embeddingLayer.computeOutputShape(x.shape)).toEqual(yExpectedShape);
    // Collect embedded output elements.
    const yData0 = y.slice([0, 0, 0, 0], [1, 1, 1, 3]).dataSync();
    const yData1 = y.slice([0, 0, 1, 0], [1, 1, 1, 3]).dataSync();
    const yData2 = y.slice([0, 0, 2, 0], [1, 1, 1, 3]).dataSync();
    const yData3 = y.slice([0, 0, 3, 0], [1, 1, 1, 3]).dataSync();
    // Collect sample embedding rows.
    const wData0 = weights.slice([0, 0], [1, 3]).dataSync();
    const wData1 = weights.slice([1, 0], [1, 3]).dataSync();
    const wData5 = weights.slice([5, 0], [1, 3]).dataSync();
    // First output should match first embedding row.
    expectArraysClose(yData0, wData0);
    // First output should *not* match second embedding row
    expect(() => {
      expectArraysClose(yData0, wData1);
    }).toThrow();
    // Second output should match 6th embedding row.
    expectArraysClose(yData1, wData5);
    // Second output should *not* match first embedding row.
    expect(() => {
      expectArraysClose(yData1, wData0);
    }).toThrow();
    // Third output should match fourth output (same embedding index);
    expectArraysClose(yData2, yData3);
  });
});
