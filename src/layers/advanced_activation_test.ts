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
 * Unit Tests for Advanced Activation Layers.
 */

import {ones, Tensor, tensor1d, tensor2d, zeros} from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import {convertPythonicToTs, convertTsToPythonic} from '../utils/serialization_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';


describeMathCPU('ReLU: Symbolic', () => {
  it('Correct output shape', () => {
    const layer = tfl.layers.reLU({});
    const x = new tfl.SymbolicTensor('float32', [2, 3, 4], null, null, null);
    const y = layer.apply(x) as tfl.SymbolicTensor;
    expect(y.shape).toEqual(x.shape);
  });

  it('Serialization round trip', () => {
    const layer = tfl.layers.reLU({maxValue: 28});
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.reLU(tsConfig);
    expect(layerPrime.getConfig().maxValue).toEqual(28);
  });
});

describeMathCPUAndGPU('ReLU: Tensor', () => {
  it('No maxValue', () => {
    const layer = tfl.layers.reLU();
    const x = tensor2d([[-100, -200], [0, 300], [200, 200]]);
    const y = layer.apply(x) as Tensor;
    expectTensorsClose(y, tensor2d([[0, 0], [0, 300], [200, 200]]));
  });

  it('Finite maxValue', () => {
    const layer = tfl.layers.reLU({maxValue: 250});
    const x = tensor1d([-100, -200, 0, 300, 200, 200]);
    const y = layer.apply(x) as Tensor;
    expectTensorsClose(y, tensor1d([0, 0, 0, 250, 200, 200]));
  });
});

describeMathCPU('PReLU: Symbolic', () => {
  it('Correct output shape: no-arg constructor', () => {
    const layer = tfl.layers.prelu();
    const x = new tfl.SymbolicTensor('float32', [2, 3, 4], null, null, null);
    const y = layer.apply(x) as tfl.SymbolicTensor;
    expect(y.shape).toEqual(x.shape);
  });

  it('Correct output shape: constructor with arg', () => {
    const layer = tfl.layers.prelu({});
    const x = new tfl.SymbolicTensor('float32', [2, 3, 4], null, null, null);
    const y = layer.apply(x) as tfl.SymbolicTensor;
    expect(y.shape).toEqual(x.shape);
  });

  it('Serialization round trip', () => {
    const layer = tfl.layers.prelu({
      alphaInitializer: 'ones',
      sharedAxes: [1, 2]
    });
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.prelu(tsConfig);
    expect(layerPrime.getConfig().sharedAxes).toEqual([1, 2]);
  });
});

describeMathCPUAndGPU('PReLU: Tensor', () => {
  it('Forward pass', () => {
    const layer = tfl.layers.prelu({alphaInitializer: 'ones'});
    const x = tensor2d([[-100, -200], [0, 300], [200, 200]]);
    const y = layer.apply(x) as Tensor;
    expectTensorsClose(y, tensor2d([[-100, -200], [0, 300], [200, 200]]));
  });

  // Reference Python code:
  // ```py
  // import keras
  // import numpy as np
  //
  // model = keras.Sequential()
  // model.add(keras.layers.Dense(
  //     3, kernel_initializer='ones', input_shape=[4]))
  // model.add(keras.layers.PReLU(alpha_initializer='ones'))
  // model.add(keras.layers.Dense(1, kernel_initializer='ones'))
  // model.compile(optimizer='sgd', loss='mean_squared_error')
  //
  // xs = -np.ones([2, 4])
  // ys = np.zeros([2, 1])
  // history = model.fit(xs, ys, batch_size=2, epochs=3)
  // print(history.history)
  // print(model.get_weights()[2])
  // ```
  it('Training: no sharedAxes', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      kernelInitializer: 'ones',
      inputShape: [4]
    }));
    model.add(tfl.layers.prelu({alphaInitializer: 'ones'}));
    model.add(tfl.layers.dense({units: 1, kernelInitializer: 'ones'}));
    model.compile({
      optimizer: 'sgd',
      loss: 'meanSquaredError'
    });

    const xs = ones([2, 4]).neg();
    const ys = zeros([2, 1]);
    const history = await model.fit(xs, ys, {
      batchSize: 2,
      epochs: 3
    });
    expect(history.history.loss.length).toEqual(3);
    expect(history.history.loss[0]).toBeCloseTo(144);
    expect(history.history.loss[1]).toBeCloseTo(0.051329);
    expect(history.history.loss[2]).toBeCloseTo(0.049144);
    expectTensorsClose(
        model.getWeights()[2], tensor1d([0.0410104, 0.0410104, 0.0410104]));
  });

  // Reference Python code:
  // ```py
  // import keras
  // import numpy as np
  //
  // model = keras.Sequential()
  // model.add(keras.layers.Dense(
  //     3, kernel_initializer='ones', input_shape=[2, 2]))
  // model.add(keras.layers.PReLU(alpha_initializer='ones', shared_axes=[1]))
  // model.add(keras.layers.Dense(1, kernel_initializer='ones'))
  // model.compile(optimizer='sgd', loss='mean_squared_error')
  //
  // xs = -np.ones([2, 2, 2])
  // ys = np.zeros([2, 2, 1])
  // history = model.fit(xs, ys, batch_size=2, epochs=3)
  // print(history.history)
  // print(model.get_weights()[2])
  // ```
  it('Training, with sharedAxes', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      kernelInitializer: 'ones',
      inputShape: [2, 2]
    }));
    model.add(tfl.layers.prelu({
      alphaInitializer: 'ones',
      sharedAxes: [1]
    }));
    model.add(tfl.layers.dense({units: 1, kernelInitializer: 'ones'}));
    model.compile({
      optimizer: 'sgd',
      loss: 'meanSquaredError'
    });

    const xs = ones([2, 2, 2]).neg();
    const ys = zeros([2, 2, 1]);
    const history = await model.fit(xs, ys, {
      batchSize: 2,
      epochs: 3
    });
    expect(history.history.loss.length).toEqual(3);
    expect(history.history.loss[0]).toBeCloseTo(36);
    expect(history.history.loss[1]).toBeCloseTo(7.408153);
    expect(history.history.loss[2]).toBeCloseTo(4.190359);
    expectTensorsClose(
        model.getWeights()[2], tensor2d([[0.648351, 0.648351, 0.648351]]));
  });
});

describeMathCPU('leakyReLU: Symbolic', () => {
  it('Correct output shape', () => {
    const layer = tfl.layers.leakyReLU({alpha: 0.1});
    const x = new tfl.SymbolicTensor('float32', [2, 3, 4], null, null, null);
    const y = layer.apply(x) as tfl.SymbolicTensor;
    expect(y.shape).toEqual(x.shape);
  });
});

describeMathCPUAndGPU('leakyReLU: Tensor', () => {
  it('alpha = default 0.3', () => {
    const layer = tfl.layers.leakyReLU();
    const x = tensor2d([[-1, -2], [0, 3]], [2, 2]);
    const y = layer.apply(x) as Tensor;
    expectTensorsClose(y, tensor2d([[-0.3, -0.6], [0, 3]], [2, 2]));
  });

  it('alpha = 0.1', () => {
    const layer = tfl.layers.leakyReLU({alpha: 0.1});
    const x = tensor2d([[-1, -2], [0, 3]], [2, 2]);
    const y = layer.apply(x) as Tensor;
    expectTensorsClose(y, tensor2d([[-0.1, -0.2], [0, 3]], [2, 2]));
  });
});

describeMathCPU('elu: Symbolic', () => {
  it('Correct output shape', () => {
    const layer = tfl.layers.elu();
    const x = new tfl.SymbolicTensor('float32', [2, 3, 4], null, null, null);
    const y = layer.apply(x) as tfl.SymbolicTensor;
    expect(y.shape).toEqual(x.shape);
  });
});

describeMathCPUAndGPU('elu: Tensor', () => {
  it('alpha = default 1.0', () => {
    const layer = tfl.layers.elu({});
    const x = tensor2d([[-1, -2], [0, 3]], [2, 2]);
    const y = layer.apply(x) as Tensor;
    expectTensorsClose(
        y, tensor2d([[Math.exp(-1) - 1, Math.exp(-2) - 1], [0, 3]], [2, 2]));
  });
});

describeMathCPU('thresholdedReLU: Symbolic', () => {
  it('Correct output shape', () => {
    const layer = tfl.layers.thresholdedReLU();
    const x = new tfl.SymbolicTensor('float32', [2, 3, 4], null, null, null);
    const y = layer.apply(x) as tfl.SymbolicTensor;
    expect(y.shape).toEqual(x.shape);
  });
});

describeMathCPUAndGPU('thresholdedReLU: Tensor', () => {
  it('theta = default 1.0', () => {
    const layer = tfl.layers.thresholdedReLU({});
    const x = tensor2d([[-1, 0], [1, 3]], [2, 2]);
    const y = layer.apply(x) as Tensor;
    expectTensorsClose(y, tensor2d([[0, 0], [0, 3]], [2, 2]));
  });
});

describeMathCPU('softmax: Symbolic', () => {
  const axisValues = [0, 1, 2, -1, null];
  for (const axis of axisValues) {
    it(`Correct output shape, axis=${axis}`, () => {
      const layer = tfl.layers.softmax({axis});
      const x = new tfl.SymbolicTensor('float32', [2, 3, 4], null, null, null);
      const y = layer.apply(x) as tfl.SymbolicTensor;
      expect(y.shape).toEqual(x.shape);
    });
  }
});

describeMathCPUAndGPU('softmax: Tensor', () => {
  it('theta = default 1.0', () => {
    const layer = tfl.layers.softmax({});
    const x = tensor2d([[0, 1], [5, 5]], [2, 2]);
    const y = layer.apply(x) as Tensor;
    expectTensorsClose(
        y,
        tensor2d(
            [[1 / (1 + Math.E), Math.E / (1 + Math.E)], [0.5, 0.5]], [2, 2]));
  });
});
