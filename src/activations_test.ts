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
// tslint:disable:max-line-length
import {scalar, tensor1d, tensor2d, tensor3d} from 'deeplearn';

import {elu, hardSigmoid, linear, relu, relu6, selu, sigmoid, softmax, softplus, softsign, tanh} from './activations';
import {describeMathCPUAndGPU, expectTensorsClose} from './utils/test_utils';
// tslint:enable

// TODO(bileschi): Here and below, these tests only check ConcreteTensor
// type.  They should be adapted to also test Symbolic Type, once it is
// available.
describeMathCPUAndGPU('linear activation', () => {
  const initVals = new Float32Array([-1, 2, 0, 4, -5, 6]);
  const expectedVals = new Float32Array([-1, 2, 0, 4, -5, 6]);
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
});

describeMathCPUAndGPU('elu activation', () => {
  const initVals = [-1, 2, 0, 4, -5, 6];
  const expectedVals = initVals.map(x => x < 0 ? Math.exp(x) - 1 : x);
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
});

describeMathCPUAndGPU('selu activation', () => {
  const initVals = [-1, 2, 0, 4, -5, 6];
  const alpha = 1.6732632423543772848170429916717;
  const scale = 1.0507009873554804934193349852946;

  const expectedVals =
      initVals.map(x => scale * (x < 0 ? (alpha * (Math.exp(x) - 1)) : x));
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
});


describeMathCPUAndGPU('relu activation', () => {
  const initVals = new Float32Array([-1, 2, 0, 4, -5, 6]);
  const expectedVals = new Float32Array([0, 2, 0, 4, 0, 6]);
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
});

describeMathCPUAndGPU('relu6 activation', () => {
  const initVals = new Float32Array([-10, -5, 0, 1, 5, 15]);
  const expectedVals = new Float32Array([0, 0, 0, 1, 5, 6]);
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
});

describeMathCPUAndGPU('sigmoid activation', () => {
  it('Scalar', () => {
    expectTensorsClose(sigmoid(scalar(0)), scalar(0.5));
  });
  it('3D', () => {
    const initVals = [-1, 2, 0, 4, -5, 6];
    const expectedVals = initVals.map(v => 1 / (1 + Math.exp(-v)));
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(sigmoid(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
});

describeMathCPUAndGPU('hardSigmoid activation', () => {
  it('Scalar', () => {
    expectTensorsClose(hardSigmoid(scalar(0)), scalar(0.5));
  });
  it('3D', () => {
    const initVals = [-1, 2, 0, 4, -5, 6];
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
});

describeMathCPUAndGPU('softplus activation', () => {
  it('Scalar', () => {
    expectTensorsClose(softplus(scalar(0)), scalar(Math.log(2)));
  });
  it('3D', () => {
    const initVals = [-1, 2, 0, 4, -5, 6];
    const expectedVals = initVals.map(v => Math.log(Math.exp(v) + 1));
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(softplus(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
});

describeMathCPUAndGPU('softsign activation', () => {
  it('Scalar', () => {
    expectTensorsClose(softsign(scalar(0)), scalar(0));
  });
  it('3D', () => {
    const initVals = [-1, 2, 0, 4, -5, 6];
    const expectedVals = initVals.map(v => v / (Math.abs(v) + 1));
    const initX = tensor3d(initVals, [1, 2, 3]);
    expectTensorsClose(softsign(initX), tensor3d(expectedVals, [1, 2, 3]));
  });
});

describeMathCPUAndGPU('tanh activation', () => {
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
});

describeMathCPUAndGPU('softmax activation', () => {
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
});
