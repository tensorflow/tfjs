/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * Unit Tests for TFJS-based EinsumDense Layer.
 */

import { memory, Tensor } from '@tensorflow/tfjs-core';

import { analyzeEinsumString, EinsumDense } from './einsum_dense';
import { Shape } from '../../keras_format/common';
import { input } from '../../exports';

declare interface EinsumDenseTestCaseArgs {
  testcaseName: string;
  equation: string;
  biasAxes: string;
  inputShape: Shape;
  outputShape: Shape;
  expectedWeightShape: Shape;
  expectedBiasShape: Shape;
  expectedOutputShape: Shape;
}

describe('EinsumDense', () => {
  const combinations: EinsumDenseTestCaseArgs[] = [
    {
      testcaseName: '_1d_end_weight',
      equation: 'ab,b->a',
      biasAxes: null,
      inputShape: [null, 32],
      outputShape: [],
      expectedWeightShape: [32],
      expectedBiasShape: null,
      expectedOutputShape: [null],
    },
    {
      testcaseName: '_2d_middle_weight',
      equation: 'ab,bc->ac',
      biasAxes: null,
      inputShape: [null, 32],
      outputShape: [64],
      expectedWeightShape: [32, 64],
      expectedBiasShape: null,
      expectedOutputShape: [null, 64],
    },
    {
        testcaseName: '_3d_bert',
        equation: 'abc,cde->abde',
        biasAxes: null,
        inputShape: [null, 1, 2],
        outputShape: [1, 3, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: null,
        expectedOutputShape: [null, 1, 3, 4],
    },
    {
        testcaseName: '_3d_3_bias',
        equation: 'abc,cde->abde',
        biasAxes: 'e',
        inputShape: [null, 1, 2],
        outputShape: [1, 3, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: [4],
        expectedOutputShape: [null, 1, 3, 4],
    },
    {
        testcaseName: '_3d_2_bias',
        equation: 'abc,cde->abde',
        biasAxes: 'd',
        inputShape: [null, 1, 2],
        outputShape: [1, 3, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: [3, 1],
        expectedOutputShape: [null, 1, 3, 4],
    },
    {
        testcaseName: '_3d_1_3_bias',
        equation: 'abc,cde->abde',
        biasAxes: 'be',
        inputShape: [null, 7, 2],
        outputShape: [7, 3, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: [7, 1, 4],
        expectedOutputShape: [null, 7, 3, 4],
    },
    {
        testcaseName: '_3d_bert_projection',
        equation: 'BFNH,NHD->BFD',
        biasAxes: null,
        inputShape: [null, 1, 2, 3],
        outputShape: [1, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: null,
        expectedOutputShape: [null, 1, 4],
    },
    {
        testcaseName: '_2d_bert',
        equation: 'abc,cd->abd',
        biasAxes: null,
        inputShape: [null, 1, 2],
        outputShape: [1, 4],
        expectedWeightShape: [2, 4],
        expectedBiasShape: null,
        expectedOutputShape: [null, 1, 4],
    },
    {
        testcaseName: '_embedding_1d',
        equation: 'i,d->id',
        biasAxes: null,
        inputShape: [null],
        outputShape: [2],
        expectedWeightShape: [2],
        expectedBiasShape: null,
        expectedOutputShape: [null, 2],
    },
    {
        testcaseName: '_xlnet_lm',
        equation: 'ibd,nd->ibn',
        biasAxes: null,
        inputShape: [null, null, 1],
        outputShape: [null, 2],
        expectedWeightShape: [2, 1],
        expectedBiasShape: null,
        expectedOutputShape: [null, null, 2],
    },
    {
        testcaseName: '_2d_precast',
        equation: '...b,bc->...c',
        biasAxes: null,
        inputShape: [null, 32],
        outputShape: [64],
        expectedWeightShape: [32, 64],
        expectedBiasShape: null,
        expectedOutputShape: [null, 64],
    },
    {
        testcaseName: '_2d_precast_elided_input_used_in_output',
        equation: '...bc,bc->...b',
        biasAxes: null,
        inputShape: [null, 32, 64],
        outputShape: [32],
        expectedWeightShape: [32, 64],
        expectedBiasShape: null,
        expectedOutputShape: [null, 32],
    },
    {
        testcaseName: '_2d_precast_multiple_elided_dims',
        equation: '...b,bc->...c',
        biasAxes: null,
        inputShape: [null, null, 32],
        outputShape: [64],
        expectedWeightShape: [32, 64],
        expectedBiasShape: null,
        expectedOutputShape: [null, null, 64],
    },
    {
        testcaseName: '_3d_precast',
        equation: '...c,cde->...de',
        biasAxes: null,
        inputShape: [null, 1, 2],
        outputShape: [3, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: null,
        expectedOutputShape: [null, 1, 3, 4],
    },
    {
        testcaseName: '_3d_precast_3_bias',
        equation: '...c,cde->...de',
        biasAxes: 'e',
        inputShape: [null, 1, 2],
        outputShape: [3, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: [4],
        expectedOutputShape: [null, 1, 3, 4],
    },
    {
        testcaseName: '_3d_precast_2_bias',
        equation: '...c,cde->...de',
        biasAxes: 'd',
        inputShape: [null, 1, 2],
        outputShape: [3, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: [3, 1],
        expectedOutputShape: [null, 1, 3, 4],
    },
    {
        testcaseName: '_3d_precast_2_3_bias',
        equation: '...c,cde->...de',
        biasAxes: 'de',
        inputShape: [null, 1, 2],
        outputShape: [3, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: [3, 4],
        expectedOutputShape: [null, 1, 3, 4],
    },
    {
        testcaseName: '_2d_postcast',
        equation: 'bc...,cd->bd...',
        biasAxes: null,
        inputShape: [null, 1, 2, 3],
        outputShape: [4],
        expectedWeightShape: [1, 4],
        expectedBiasShape: null,
        expectedOutputShape: [null, 4, 2, 3],
    },
    {
        testcaseName: '_3d_postcast',
        equation: 'bc...,cde->bde...',
        biasAxes: null,
        inputShape: [null, 1, 2],
        outputShape: [3, 4],
        expectedWeightShape: [1, 3, 4],
        expectedBiasShape: null,
        expectedOutputShape: [null, 3, 4, 2],
    },
    {
        testcaseName: '_3d_postcast_1_bias',
        equation: 'bc...,cde->bde...',
        biasAxes: 'd',
        inputShape: [null, 1, 2],
        outputShape: [3, 4],
        expectedWeightShape: [1, 3, 4],
        expectedBiasShape: [3, 1, 1],
        expectedOutputShape: [null, 3, 4, 2],
    },
    {
        testcaseName: '_3d_postcast_2_bias',
        equation: 'bc...,cde->bde...',
        biasAxes: 'e',
        inputShape: [null, 1, 2],
        outputShape: [3, 4],
        expectedWeightShape: [1, 3, 4],
        expectedBiasShape: [4, 1],
        expectedOutputShape: [null, 3, 4, 2],
    },
    {
        testcaseName: '_3d_postcast_1_2_bias',
        equation: 'bc...,cde->bde...',
        biasAxes: 'de',
        inputShape: [null, 1, 2],
        outputShape: [3, 4],
        expectedWeightShape: [1, 3, 4],
        expectedBiasShape: [3, 4, 1],
        expectedOutputShape: [null, 3, 4, 2],
    },
  ];

  function testWeightShape(combo: EinsumDenseTestCaseArgs) {
    it(`${combo.testcaseName} weight shape`, () => {
      const [weightShape, biasShape, _] = analyzeEinsumString(
        combo.equation, combo.biasAxes, combo.inputShape, combo.outputShape
      );
      expect(weightShape).toEqual(combo.expectedWeightShape);
      expect(biasShape).toEqual(combo.expectedBiasShape);
    });
  }

  function testLayerCreation(combo: EinsumDenseTestCaseArgs) {
    it(`${combo.testcaseName} layer creation`, () => {
      const nonBatchInputShape = combo.inputShape.slice(1);
      const inputTensor = input({shape: nonBatchInputShape});

      const layer = new EinsumDense({
        equation: combo.equation,
        biasAxes: combo.biasAxes,
        outputShape: combo.outputShape,
      });
      const outputTensor = layer.apply(inputTensor) as Tensor;

      expect(layer.kernel.shape).toEqual(combo.expectedWeightShape);
      if (combo.expectedBiasShape === null) {
        expect(layer.bias).toBeNull();
      } else {
        expect(layer.bias.shape).toEqual(combo.expectedBiasShape);
      }
      expect(outputTensor.shape).toEqual(combo.expectedOutputShape);
    });
  }

  for (const combo of combinations) {
    testWeightShape(combo);
    testLayerCreation(combo);
  }

  it('Does not leak memory', () => {
    const combo = combinations[0];
    const layer = new EinsumDense({
      equation: combo.equation,
      biasAxes: combo.biasAxes,
      outputShape: combo.outputShape,
    });
    const nonBatchInputShape = combo.inputShape.slice(1);
    const inputTensor = input({shape: nonBatchInputShape});

    const numTensors = memory().numTensors;
    layer.apply(inputTensor);

    expect(memory().numTensors).toEqual(numTensors + 1);
  });

  // TODO(pforderique): Test serialization.
});
