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
import {scalar, tensor1d, tensor2d, tensor3d} from '@tensorflow/tfjs-core';

import {Elu, HardSigmoid, Linear, Relu, Relu6, Selu, Sigmoid, Softmax, Softplus, Softsign, Tanh} from './activations';
import {describeMathCPUAndGPU, expectNoLeakedTensors, expectTensorsClose} from './utils/test_utils';


describeMathCPUAndGPU('linear activation', () => {
  const initVals = new Float32Array([-1, 2, 0, 4, -5, 6]);
  const expectedVals = new Float32Array([-1, 2, 0, 4, -5, 6]);
  const linear = new Linear().apply;
  it('1D', () => {
    const initX = tensor1d(initVals);
    expectTensorsClose(linear(initX), tensor1d(expectedVals));
  });
  it('2D', () => {
    const initX = tensor2d(initVals, [2, 3]);
    expectTensorsClose(linear(initX), tensor2d(expectedVals, [2, 3]));
  });
  it('3D', () => {
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(linear(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
  it('Does not leak', () => {
    const initX = tensor1d(initVals);
    expectNoLeakedTensors(() => linear(initX), 0);
  });
});

describeMathCPUAndGPU('elu activation', () => {
  const initVals = [-1, 2, 0, 4, -5, 6];
  const expectedVals = initVals.map(x => x < 0 ? Math.exp(x) - 1 : x);
  const elu = new Elu().apply;
  it('1D', () => {
    const initX = tensor1d(initVals);
    expectTensorsClose(elu(initX), tensor1d(expectedVals));
  });
  it('2D', () => {
    const initX = tensor2d(initVals, [2, 3]);
    expectTensorsClose(elu(initX), tensor2d(expectedVals, [2, 3]));
  });
  it('3D', () => {
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(elu(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
  it('Does not leak', () => {
    const initX = tensor1d(initVals);
    expectNoLeakedTensors(() => elu(initX), 1);
  });
});

describeMathCPUAndGPU('selu activation', () => {
  const initVals = [-1, 2, 0, 4, -5, 6];
  const alpha = 1.6732632423543772848170429916717;
  const scale = 1.0507009873554804934193349852946;

  const expectedVals =
      initVals.map(x => scale * (x < 0 ? (alpha * (Math.exp(x) - 1)) : x));
  const selu = new Selu().apply;

  it('1D', () => {
    const initX = tensor1d(initVals);
    expectTensorsClose(selu(initX), tensor1d(expectedVals));
  });
  it('2D', () => {
    const initX = tensor2d(initVals, [2, 3]);
    expectTensorsClose(selu(initX), tensor2d(expectedVals, [2, 3]));
  });
  it('3D', () => {
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(selu(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
  it('Does not leak', () => {
    const initX = tensor1d(initVals);
    expectNoLeakedTensors(() => selu(initX), 1);
  });
});


describeMathCPUAndGPU('relu activation', () => {
  const initVals = new Float32Array([-1, 2, 0, 4, -5, 6]);
  const expectedVals = new Float32Array([0, 2, 0, 4, 0, 6]);
  const relu = new Relu().apply;
  it('1D', () => {
    const initX = tensor1d(initVals);
    expectTensorsClose(relu(initX), tensor1d(expectedVals));
  });
  it('2D', () => {
    const initX = tensor2d(initVals, [2, 3]);
    expectTensorsClose(relu(initX), tensor2d(expectedVals, [2, 3]));
  });
  it('3D', () => {
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(relu(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
  it('Does not leak', () => {
    const initX = tensor1d(initVals);
    expectNoLeakedTensors(() => relu(initX), 1);
  });
});

describeMathCPUAndGPU('relu6 activation', () => {
  const initVals = new Float32Array([-10, -5, 0, 1, 5, 15]);
  const expectedVals = new Float32Array([0, 0, 0, 1, 5, 6]);
  const relu6 = new Relu6().apply;
  it('1D', () => {
    const initX = tensor1d(initVals);
    expectTensorsClose(relu6(initX), tensor1d(expectedVals));
  });
  it('2D', () => {
    const initX = tensor2d(initVals, [2, 3]);
    expectTensorsClose(relu6(initX), tensor2d(expectedVals, [2, 3]));
  });
  it('3D', () => {
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(relu6(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
  it('Does not leak', () => {
    const initX = tensor1d(initVals);
    expectNoLeakedTensors(() => relu6(initX), 1);
  });
});

describeMathCPUAndGPU('sigmoid activation', () => {
  const sigmoid = new Sigmoid().apply;
  const initVals = [-1, 2, 0, 4, -5, 6];
  it('Scalar', () => {
    expectTensorsClose(sigmoid(scalar(0)), scalar(0.5));
  });
  it('3D', () => {
    const expectedVals = initVals.map(v => 1 / (1 + Math.exp(-v)));
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(sigmoid(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
  it('Does not leak', () => {
    const initX = tensor1d(initVals);
    expectNoLeakedTensors(() => sigmoid(initX), 1);
  });
});

describeMathCPUAndGPU('hardSigmoid activation', () => {
  const hardSigmoid = new HardSigmoid().apply;
  const initVals = [-1, 2, 0, 4, -5, 6];
  it('Scalar', () => {
    expectTensorsClose(hardSigmoid(scalar(0)), scalar(0.5));
  });
  it('3D', () => {
    const expectedVals = initVals.map(v => {
      const y = 0.2 * v + 0.5;
      if (y > 1) {
        return 1;
      } else if (y < 0) {
        return 0;
      } else {
        return y;
      }
    });
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(hardSigmoid(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
  it('Does not leak', () => {
    const initX = tensor1d(initVals);
    expectNoLeakedTensors(() => hardSigmoid(initX), 1);
  });
});

describeMathCPUAndGPU('softplus activation', () => {
  const softplus = new Softplus().apply;
  const initVals = [-1, 2, 0, 4, -5, 6];
  it('Scalar', () => {
    expectTensorsClose(softplus(scalar(0)), scalar(Math.log(2)));
  });
  it('3D', () => {
    const expectedVals = initVals.map(v => Math.log(Math.exp(v) + 1));
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(softplus(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
  it('Does not leak', () => {
    const initX = tensor1d(initVals);
    expectNoLeakedTensors(() => softplus(initX), 1);
  });
});

describeMathCPUAndGPU('softsign activation', () => {
  const softsign = new Softsign().apply;
  const initVals = [-1, 2, 0, 4, -5, 6];
  it('Scalar', () => {
    expectTensorsClose(softsign(scalar(0)), scalar(0));
  });
  it('3D', () => {
    const expectedVals = initVals.map(v => v / (Math.abs(v) + 1));
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(softsign(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
  it('Does not leak', () => {
    const initX = tensor1d(initVals);
    expectNoLeakedTensors(() => softsign(initX), 1);
  });
});

describeMathCPUAndGPU('tanh activation', () => {
  const tanh = new Tanh().apply;
  const initVals = [-1, 2, 0, 4, -5, 6];
  const expectedVals = initVals.map(x => Math.tanh(x));
  it('1D', () => {
    const initX = tensor1d(initVals);
    expectTensorsClose(tanh(initX), tensor1d(expectedVals));
  });
  it('2D', () => {
    const initX = tensor2d(initVals, [2, 3]);
    expectTensorsClose(tanh(initX), tensor2d(expectedVals, [2, 3]));
  });
  it('3D', () => {
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(tanh(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
  it('Does not leak', () => {
    const initX = tensor1d(initVals);
    expectNoLeakedTensors(() => tanh(initX), 1);
  });
});

describeMathCPUAndGPU('softmax activation', () => {
  const softmax = new Softmax().apply;
  // Setup: Array with initial values.
  // Execute: Softmax on the last dimension.
  // Expect: Output array matches size and approximate expected values.
  it('1D', () => {
    const initVals = new Float32Array([0, 1, 3, 9]);
    const expectedVals = new Float32Array([0.000, 0.000, 0.002, 0.997]);
    const initX = tensor1d(initVals);
    expectTensorsClose(softmax(initX), tensor1d(expectedVals));
  });
  it('1D all equal', () => {
    const initVals = new Float32Array([-1, -1, -1, -1]);
    const expectedVals = new Float32Array([0.25, 0.25, 0.25, 0.25]);
    const initX = tensor1d(initVals);
    expectTensorsClose(softmax(initX), tensor1d(expectedVals));
  });
  it('2D', () => {
    const initVals = new Float32Array([0, 1, 3, 9, 0, 1, 3, 9]);
    const expectedVals = new Float32Array(
        [0.000, 0.000, 0.002, 0.997, 0.000, 0.000, 0.002, 0.997]);
    const initX = tensor2d(initVals, [2, 4]);
    expectTensorsClose(softmax(initX), tensor2d(expectedVals, [2, 4]));
  });
  it('3D', () => {
    const initVals = new Float32Array([0, 1, 3, 9, 0, 1, 3, 9]);
    const expectedVals = new Float32Array(
        [0.000, 0.000, 0.002, 0.997, 0.000, 0.000, 0.002, 0.997]);
    const initX = tensor3d(initVals, [1, 2, 4]);
    expectTensorsClose(softmax(initX), tensor3d(expectedVals, [1, 2, 4]));
  });
  it('Does not leak', () => {
    const initVals = new Float32Array([0, 1, 3, 9]);
    const initX = tensor1d(initVals);
    expectNoLeakedTensors(() => softmax(initX), 1);
  });
});
