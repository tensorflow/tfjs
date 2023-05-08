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
 * Unit Tests for Noise Layers.
 */

import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';
import * as tfl from '../index';
import {ones, Tensor} from '@tensorflow/tfjs-core';
import {getExactlyOneTensor} from '../utils/types_utils';

describeMathCPU('GaussianNoise: Symbolic', () => {
  const stddevs = [0, 1, 0.5];
  const symbolicInputs = [
    new tfl.SymbolicTensor('float32', [10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [null, 4], null, [], null),
  ];

  for (const stddev of stddevs) {
    for (const symbolicInput of symbolicInputs) {
      const testTitle = `std=${stddev}; ` +
        `input shape=${JSON.stringify(symbolicInput.shape)}`;
      it(testTitle, () => {
        const gaussianNoiseLayer = tfl.layers.gaussianNoise({stddev});
        const output =
            gaussianNoiseLayer.apply(symbolicInput) as tfl.SymbolicTensor;
        expect(output.dtype).toEqual(symbolicInput.dtype);
        expect(output.shape).toEqual(symbolicInput.shape);
        expect(output.sourceLayer).toEqual(gaussianNoiseLayer);
        expect(output.inputs).toEqual([symbolicInput]);
      });
    }
  }
});

describeMathCPUAndGPU('GaussianNoise: Tensor', () => {
  it('GaussianNoise: Predict', () => {
    const input = ones([2, 2]);
    const gaussianNoiseLayer = tfl.layers.gaussianNoise({stddev: 1});
    let output = gaussianNoiseLayer.apply(input, {training: false}) as Tensor;
    output = getExactlyOneTensor(output);
    expectTensorsClose(input, output);
  });

  it('GaussianNoise: Train', () => {
    const input = ones([2, 2]);
    const gaussianNoiseLayer = tfl.layers.gaussianNoise({stddev: 1});
    let output = gaussianNoiseLayer.apply(input, {training: true}) as Tensor;
    output = getExactlyOneTensor(output);
    const diff = output.sub(input).abs().max();
    expect(diff.dataSync()).toBeGreaterThan(0);
    expect(output.shape).toEqual(input.shape);
    expect(output.dtype).toEqual(input.dtype);
  });

  it('GaussianNoise: Successive Call', () => {
    const training = true;
    const inputA = ones([2, 2]);
    const gaussianNoiseLayer = tfl.layers.gaussianNoise({stddev: 1});
    let outputA = gaussianNoiseLayer.apply(inputA, {training}) as Tensor;
    outputA = getExactlyOneTensor(outputA);
    const inputB = ones([2, 2]);
    let outputB = gaussianNoiseLayer.apply(inputB, {training}) as Tensor;
    outputB = getExactlyOneTensor(outputA);
    expectTensorsClose(outputA, outputB);
  });
});

describeMathCPU('GaussianDropout: Symbolic', () => {
  const rates = [0, 1, 0.5];
  const symbolicInputs = [
    new tfl.SymbolicTensor('float32', [10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [null, 4], null, [], null),
  ];

  for (const rate of rates) {
    for (const symbolicInput of symbolicInputs) {
      const testTitle = `dropoutRate=${rate}; ` +
        `input shape=${JSON.stringify(symbolicInput.shape)}`;
      it(testTitle, () => {
        const gaussianDropout = tfl.layers.gaussianDropout({rate});
        const output =
          gaussianDropout.apply(symbolicInput) as tfl.SymbolicTensor;
        expect(output.dtype).toEqual(symbolicInput.dtype);
        expect(output.shape).toEqual(symbolicInput.shape);
        expect(output.sourceLayer).toEqual(gaussianDropout);
        expect(output.inputs).toEqual([symbolicInput]);
      });
    }
  }
});

describeMathCPUAndGPU('GaussianDropout: Tensor', () => {
  it('GaussianDropout: Predict', () => {
    const input = ones([1, 2]);
    const gaussianDropoutLayer = tfl.layers.gaussianDropout({rate: 0.5});
    let output = gaussianDropoutLayer.apply(input, {training: false}) as Tensor;
    output = getExactlyOneTensor(output);
    expectTensorsClose(input, output);
  });

  it('GaussianDropout: Train', () => {
    const input = ones([1, 2]);
    const gaussianDropoutLayer = tfl.layers.gaussianDropout({rate: 0.5});
    let output = gaussianDropoutLayer.apply(input, {training: true}) as Tensor;
    output = getExactlyOneTensor(output);
    const diff = output.sub(input).abs().max();
    expect(diff.dataSync()).toBeGreaterThan(0);
    expect(output.shape).toEqual(input.shape);
    expect(output.dtype).toEqual(input.dtype);
  });

  it('GaussianDropout: Successive Call', () => {
    const training = true;
    const inputA = ones([1, 2]);
    const gaussianDropoutLayer = tfl.layers.gaussianDropout({rate: 0.5});
    let outputA = gaussianDropoutLayer.apply(inputA, {training}) as Tensor;
    outputA = getExactlyOneTensor(outputA);
    const inputB = ones([1, 2]);
    let outputB = gaussianDropoutLayer.apply(inputB, {training}) as Tensor;
    outputB = getExactlyOneTensor(outputA);
    expectTensorsClose(outputA, outputB);
  });
});

describeMathCPU('AlphaDropout: Symbolic', () => {
  const rates = [0, 1, 0.5];
  const symbolicInputs = [
    new tfl.SymbolicTensor('float32', [10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [null, 4], null, [], null),
  ];

  for (const rate of rates) {
    for (const symbolicInput of symbolicInputs) {
      const testTitle = `dropoutRate=${rate}; ` +
        `input shape=${JSON.stringify(symbolicInput.shape)}`;
      it(testTitle, () => {
        const alphaDropout = tfl.layers.alphaDropout({rate});
        const output = alphaDropout.apply(symbolicInput) as tfl.SymbolicTensor;
        expect(output.dtype).toEqual(symbolicInput.dtype);
        expect(output.shape).toEqual(symbolicInput.shape);
        expect(output.sourceLayer).toEqual(alphaDropout);
        expect(output.inputs).toEqual([symbolicInput]);
      });
    }
  }
});

describeMathCPUAndGPU('AlphaDropout: Tensor', () => {
  it('AlphaDropout: Predict', () => {
    const input = ones([1, 2]);
    const alphaDropoutLayer = tfl.layers.alphaDropout({rate: 0.5});
    let output = alphaDropoutLayer.apply(input, {training: false}) as Tensor;
    output = getExactlyOneTensor(output);
    expectTensorsClose(input, output);
  });

  it('AlphaDropout: Train', () => {
    const input = ones([1, 2]);
    const alphaDropoutLayer = tfl.layers.alphaDropout({rate: 0.5});
    let output = alphaDropoutLayer.apply(input, {training: true}) as Tensor;
    output = getExactlyOneTensor(output);
    const diff = output.sub(input).abs().max();
    expect(diff.dataSync()).toBeGreaterThan(0);
    expect(output.shape).toEqual(input.shape);
    expect(output.dtype).toEqual(input.dtype);
  });

  it('AlphaDropout: Successive Call', () => {
    const training = true;
    const inputA = ones([1, 2]);
    const alphaDropoutLayer = tfl.layers.alphaDropout({rate: 0.5});
    let outputA = alphaDropoutLayer.apply(inputA, {training}) as Tensor;
    outputA = getExactlyOneTensor(outputA);
    const inputB = ones([1, 2]);
    let outputB = alphaDropoutLayer.apply(inputB, {training}) as Tensor;
    outputB = getExactlyOneTensor(outputA);
    expectTensorsClose(outputA, outputB);
  });
});
