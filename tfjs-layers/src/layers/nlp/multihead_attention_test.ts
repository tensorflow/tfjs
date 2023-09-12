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
 * Unit Tests for MultiHeadAttention layer.
 */

import { Tensor, memory, ones, randomUniform, randomUniformInt, tensor, tensor2d } from '@tensorflow/tfjs-core';

import { TruncatedNormal } from '../../initializers';
import { input } from '../../exports';
import { Shape } from '../../keras_format/common';
import { MultiHeadAttention } from './multihead_attention';
import { describeMathCPU, describeMathCPUAndGPU, expectTensorsClose, expectTensorsNotClose } from '../../utils/test_utils';
import { Embedding } from '../embeddings';

describeMathCPUAndGPU('MultiHeadAttention', () => {

  describe('Non Masked Attention', () => {
    interface NonMaskedAttentionArgs {
      testcaseName: string;
      valueDim: number;
      outputShape: Shape;
      outputDims: Shape;
    }
    /**
     * Test that the attention layer can be created without a mask tensor.
     */
    function testNonMaskedAttention(
      {testcaseName, valueDim, outputShape, outputDims}: NonMaskedAttentionArgs
    ) {
      it(`${testcaseName} non masked attention`, () => {
        const testLayer = new MultiHeadAttention({
          numHeads: 12,
          keyDim: 64,
          valueDim,
          outputShape,
        });
        // Create a 3-dimensional input (the first dimension is implicit).
        const query = input({shape: [40, 80]});
        const value = input({shape: [20, 80]});
        const output = testLayer.apply(query, {value}) as Tensor;
        expect(output.shape).toEqual([null].concat(outputDims));
      });
    }

    const params: NonMaskedAttentionArgs[] = [
      {
        testcaseName: 'key value same proj',
        valueDim: null,
        outputShape: null,
        outputDims: [40, 80],
      },
      {
        testcaseName: 'key value different proj',
        valueDim: 32,
        outputShape: [60],
        outputDims: [40, 60],
      }
    ];
    for (const param of params) {
      testNonMaskedAttention(param);
    }
  });

  // Test with one input (self-attention) and no mask tensor.
  it('non masked self attention', () => {
    const testLayer = new MultiHeadAttention({numHeads: 12, keyDim: 64});
    // Create a 3-dimensional input (the first dimension is implicit).
    const query = input({shape: [40, 80]});
    const output = testLayer.apply(query, {value: query}) as Tensor;
    expect(output.shape).toEqual([null, 40, 80]);
  });

  // Test attention outputs with coefficients.
  it('attention scores', () => {
    const testLayer = new MultiHeadAttention({numHeads: 12, keyDim: 64});
    const query = ones([1, 40, 80]);
    const [output, coef] =
      testLayer.callAndReturnAttentionScores(query, {value: query});
    expect(output.shape).toEqual([1, 40, 80]);
    expect(coef.shape).toEqual([1, 12, 40, 40]);
  });

  // Test attention outputs with coefficients.
  it('attention scores with values', () => {
    const testLayer = new MultiHeadAttention({numHeads: 12, keyDim: 64});
    const query = ones([1, 40, 80]);
    const value = ones([1, 60, 80]);
    const [output, coef] =
      testLayer.callAndReturnAttentionScores(query, {value});
    expect(output.shape).toEqual([1, 40, 80]);
    expect(coef.shape).toEqual([1, 12, 40, 60]);
  });

  describe('Masked Attention', () => {
    interface MaskedAttentionArgs {
      testcaseName: string;
      useBias: boolean;
    }
    /**
     * Test with a mask tensor.
     */
    function testMaskedAttention({testcaseName, useBias}: MaskedAttentionArgs) {
      it(`${testcaseName}`, () => {
        const testLayer = new MultiHeadAttention({
          numHeads: 2,
          keyDim: 2,
          useBias,
        });
        // Create a 3-dimensional input (the first dimension is implicit).
        const batchSize = 4;
        const query = randomUniform([batchSize, 4, 8]);
        const value = randomUniform([batchSize, 2, 8]);

        // Invoke the data with a random set of mask data. This should mask at
        // least one element.
        const maskData = randomUniformInt([batchSize, 4, 2], 0, 2);
        const maskedOutputData = testLayer.call(
          query, {value, attentionMask: maskData});

        // Invoke the same data, but with a null mask (where no elements are
        // masked).
        const nullMaskData = ones([batchSize, 4, 2]);
        const unmaskedOutputData = testLayer.call(
          query, {value, attentionMask: nullMaskData});

        expectTensorsNotClose(maskedOutputData, unmaskedOutputData);

        if (useBias) {
          expect(testLayer._queryDense.trainableWeights.length).toEqual(2);
          expect(testLayer._outputDense.trainableWeights.length).toEqual(2);
        } else {
          expect(testLayer._queryDense.trainableWeights.length).toEqual(1);
          expect(testLayer._outputDense.trainableWeights.length).toEqual(1);
        }
      });
    }
    const params: MaskedAttentionArgs[] = [
      {
        testcaseName: 'with bias',
        useBias: true,
      },
      {
        testcaseName: 'no bias',
        useBias: false,
      }
    ];
    for (const param of params) {
      testMaskedAttention(param);
    }
  });

  // Test with a specified initializer.
  it('initializers', () => {
    const testLayer = new MultiHeadAttention({
      numHeads: 12,
      keyDim: 64,
      kernelInitializer: new TruncatedNormal({stddev: 0.02}),
    });
    const query = ones([1, 40, 80]);
    const output = testLayer.call(query, {value: query});
    expect(output.shape).toEqual([1, 40, 80]);

    // Make sure the sub layers have different kernel init value, and not
    // reusing the initializers.
    const queryKernel = testLayer._queryDense.kernel.read();
    const keyKernel = testLayer._keyDense.kernel.read();
    const valueKernel = testLayer._valueDense.kernel.read();
    const outputKernel = testLayer._outputDense.kernel.read();

    expectTensorsNotClose(queryKernel, keyKernel, 1e-6);
    expectTensorsNotClose(queryKernel, valueKernel, 1e-6);
    expectTensorsNotClose(queryKernel, outputKernel, 1e-6);
  });

  it('dropout', () => {
    const testLayer = new MultiHeadAttention({
      numHeads: 2,
      keyDim: 2,
      dropout: 0.5,
    });
    const fromData = ones([32, 4, 8]);
    const toData = ones([32, 2, 8]);

    const trainOut = testLayer.call(fromData, {value: toData, training: true});
    const testOut = testLayer.call(fromData, {value: toData, training: false});

    expectTensorsNotClose(trainOut, testOut);
  });

  describe('Causal Mask Value', () => {
    let testLayer: MultiHeadAttention;
    let maskedQuery: Tensor;
    let maskedValue: Tensor;
    let mask: Tensor;

    beforeEach(() => {
      testLayer = new MultiHeadAttention({numHeads: 2, keyDim: 2});
      const query = tensor2d([
        [1, 2, 3, 0, 0], [3, 3, 1, 1, 2], [1, 0, 0, 0, 0]
      ]);
      const maskedQueryLayer = new Embedding(
        {inputDim: 4, outputDim: 8, maskZero: true});
      maskedQuery = maskedQueryLayer.apply(query) as Tensor;
      const value = tensor2d([[5, 4, 0], [3, 0, 0], [2, 1, 1]]);
      maskedValue = new Embedding(
        {inputDim: 6, outputDim: 8, maskZero: true}).apply(value) as Tensor;

      mask = tensor([
        Array<boolean[]>(3).fill([true, true, false]).concat(
          Array<boolean[]>(2).fill([false, false, false])),
        Array<boolean[]>(5).fill([true, false, false]),
        [[true, true, true]].concat(
          Array<boolean[]>(4).fill([false, false, false]))
      ]);
    });

    /**
     * Test that the value and causal masks are taken into account.
     */
    it('causal', () => {
      const output = testLayer.call(
        maskedQuery, {value: maskedValue, useCausalMask: true});

      mask = mask.logicalAnd(tensor([
        [[true, false, false], [true, true, false]].concat(
          [[true, true, true], [true, true, true], [true, true, true]])
      ]));

      const outputWithManualMask = testLayer.call(
        maskedQuery, {value: maskedValue, attentionMask: mask});

      expectTensorsClose(output, outputWithManualMask);
    });

    it('not_causal', () => {
      const output = testLayer.call(
        maskedQuery, {value: maskedValue, useCausalMask: false});

      const outputWithManualMask = testLayer.call(
        maskedQuery, {value: maskedValue, attentionMask: mask});

      expectTensorsClose(output, outputWithManualMask);
    });
  });

  describe('Compute Output Shape', () => {
    interface ComputeOutputShapeArgs {
      testcaseName: string;
      queryDims: Shape;
      valueDims: Shape;
      keyDims?: Shape;
      outputShape: Shape;
    }
    /**
     * Test computed shape is equal to the layer output's shape.
     */
    function testComputeOutputShape({
      testcaseName, queryDims, valueDims, keyDims, outputShape,
    }: ComputeOutputShapeArgs) {
      it(testcaseName, () => {
        const testLayer = new MultiHeadAttention({
          numHeads: 2,
          keyDim: 2,
          valueDim: 2,
          outputShape
        });
        const batchSize = 1;

        const queryShape = [batchSize].concat(queryDims);
        const valueShape = [batchSize].concat(valueDims);
        const keyShape = keyDims ? [batchSize].concat(keyDims) : null;

        const query = randomUniform(queryShape);
        const value = randomUniform(valueShape);
        const key = keyShape ? randomUniform(keyShape) : null;

        const output = testLayer.call(query, {value, key});
        const computedOutputShape = testLayer.computeOutputShape(
          [queryShape, valueShape, keyShape]);

        expect(output.shape).toEqual(computedOutputShape);
      });
    }
    const params: ComputeOutputShapeArgs[] = [
      {
        testcaseName: 'without_key_same_proj',
        queryDims: [40, 80],
        valueDims: [20, 80],
        keyDims: null,
        outputShape: null
      },
      {
        testcaseName: 'with_key_same_proj',
        queryDims: [40, 80],
        valueDims: [20, 80],
        keyDims: [20, 30],
        outputShape: null
      },
      {
        testcaseName: 'wihtout_key_different_proj',
        queryDims: [40, 80],
        valueDims: [20, 80],
        keyDims: null,
        outputShape: [30, 40]
      },
      {
        testcaseName: 'with_key_different_proj',
        queryDims: [40, 80],
        valueDims: [20, 80],
        keyDims: [20, 30],
        outputShape: [15, 50]
      },
    ];
    for (const param of params) {
      testComputeOutputShape(param);
    }
  });

  describe('Compute Output Shape Raises Error', () => {
    interface ComputeOutputShapeErrorArgs {
      testcaseName: string;
      queryShape: Shape;
      valueShape: Shape;
      keyShape?: Shape;
    }
    /**
     * Test dimension mismatches.
     */
    function testComputeOutputShapeError({
      testcaseName, queryShape, valueShape, keyShape,
    }: ComputeOutputShapeErrorArgs) {
      it(testcaseName, () => {
        const testLayer = new MultiHeadAttention({
          numHeads: 4,
          keyDim: 2,
          valueDim: 2,
        });

        expect(() => testLayer.computeOutputShape(
          [queryShape, valueShape, keyShape])).toThrow();
      });
    }
    const params: ComputeOutputShapeErrorArgs[] = [
      {
        testcaseName: 'query_value_dim_mismatch',
        queryShape: [null, 40, 80],
        valueShape: [null, 20, 70],
        keyShape: null
      },
      {
        testcaseName: 'key_value_dim_mismatch',
        queryShape: [null, 40, 80],
        valueShape: [null, 20, 80],
        keyShape: [null, 10, 70],
      },
      {
        testcaseName:'key_value_dim_mismatch_high_dim',
        queryShape: [null, 40, 20, 30, 80],
        valueShape: [null, 10, 10, 50, 80],
        keyShape: [null, 10, 15, 50, 20],
      },
    ];
    for (const param of params) {
      testComputeOutputShapeError(param);
    }
  });

  it('does not leak memory', () => {
    const layer = new MultiHeadAttention({numHeads: 2, keyDim: 2});
    const query = ones([1, 4, 8]);
    // Initial call that builds sublayers and necessary tensors.
    layer.call(query, {value: query});

    const numTensors = memory().numTensors;
    layer.call(query, {value: query});

    expect(memory().numTensors).toEqual(numTensors + 1);
  });
  // TODO(pforderique): Test serialization.
});

describeMathCPU('High Dimensional Attention', () => {
  interface HighDimAttentionArgs {
    testcaseName: string;
    qDims: Shape;
    vDims: Shape;
    maskDims: Shape;
    attentionAxes: number[];
  }
  /**
   * Test with high dimensional inputs.
   */
  function testHighDimAttention({
    testcaseName, qDims, vDims, maskDims, attentionAxes,
  }: HighDimAttentionArgs) {
    it(testcaseName, () => {
      const testLayer = new MultiHeadAttention({
        numHeads: 2, keyDim: 2, attentionAxes,
      });
      const batchSize = 3;
      const hiddenSize = 8;
      // Generate data for the input (non-mask) tensors.
      const queryShape = [batchSize].concat(qDims).concat(hiddenSize);
      const valueShape = [batchSize].concat(vDims).concat(hiddenSize);
      const maskShape = [batchSize].concat(maskDims);
      const query = randomUniform(queryShape, 0, 10);
      const value = randomUniform(valueShape, 0, 10);

      // Invoke the data with a random set of mask data. This should mask at
      // least one element.
      const maskData = randomUniformInt(maskShape, 0, 2).asType('bool');

      // Invoke the same data, but with a null mask (where no elements are
      // masked).
      const nullMaskData = ones(maskShape);

      // Because one data is masked and one is not, the outputs should not be
      // the same.

      const outputWithMask = testLayer.call(
        query, {value, attentionMask: maskData});
      const outputWithNullMask = testLayer.call(
        query, {value, attentionMask: nullMaskData});

      expectTensorsNotClose(outputWithMask, outputWithNullMask);
    });
  }
  const params: HighDimAttentionArgs[] = [
    {
      testcaseName: '4d_inputs_1freebatch_mask2',
      qDims: [3, 4],
      vDims: [3, 2],
      maskDims: [4, 2],
      attentionAxes: [2],
    },
    {
      testcaseName: '4d_inputs_1freebatch_mask3',
      qDims: [3, 4],
      vDims: [3, 2],
      maskDims: [3, 4, 2],
      attentionAxes: [2],
    },
    {
      testcaseName: '4d_inputs_1freebatch_mask4',
      qDims: [3, 4],
      vDims: [3, 2],
      maskDims: [3, 2, 4, 2],
      attentionAxes: [2],
    },
    {
      testcaseName: '4D_inputs_2D_attention',
      qDims: [3, 4],
      vDims: [3, 2],
      maskDims: [3, 4, 3, 2],
      attentionAxes: [1, 2],
    },
    {
      testcaseName: '5D_inputs_2D_attention',
      qDims: [5, 3, 4],
      vDims: [5, 3, 2],
      maskDims: [3, 4, 3, 2],
      attentionAxes: [2, 3],
    },
    {
      testcaseName: '5D_inputs_2D_attention_fullmask',
      qDims: [5, 3, 4],
      vDims: [5, 3, 2],
      maskDims: [5, 3, 4, 3, 2],
      attentionAxes: [2, 3],
    },
  ];
  for (const param of params) {
    testHighDimAttention(param);
  }
});

class SubclassAttention extends MultiHeadAttention {
  protected override buildAttention(qkvRank: number) {}

  protected override computeAttention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attentionMask?: Tensor,
    training?: boolean
  ): [Tensor, Tensor] {
    return [value, null];
  }
}

describe('AttentionSubclass', () => {
  // Test with a specified initializer.
  it('initializer', () => {
    const testLayer = new SubclassAttention({numHeads: 12, keyDim: 64});
    // Create a 3-dimensional input.
    const query = ones([1, 40, 80]);
    const output = testLayer.call(query, {value: query});

    expect(output.shape).toEqual([1, 40, 80]);
  });
});
