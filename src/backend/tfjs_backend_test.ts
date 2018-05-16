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
 * Unit tests for the TF.js core backend of TF.js Layers.
 */

// tslint:disable:max-line-length
import * as tfc from '@tensorflow/tfjs-core';
import {DataType, Scalar, scalar, Tensor, memory, tensor1d, tensor2d, tensor3d, tensor4d, zeros} from '@tensorflow/tfjs-core';

import {SymbolicTensor} from '../types';
import {LayerVariable} from '../variables';
import {unique} from '../utils/generic_utils';
import {range} from '../utils/math_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose, expectNoLeakedTensors} from '../utils/test_utils';

import * as K from './tfjs_backend';

// tslint:enable:max-line-length

describe('TensorMath', () => {
  it('Setting and getting backend', () => {
    // Default deeplearn.js backend is WebGL (GPU).
    const originalBackend = K.getBackend();
    expect(originalBackend).toEqual('webgl');

    K.setBackend('cpu');
    expect(K.getBackend()).toEqual('cpu');
  });
});

describe('shape', () => {
  it('Scalar', () => {
    const x = zeros([]);
    expect(K.shape(x)).toEqual([]);
  });

  it('Tensor1D', () => {
    const x = zeros([3]);
    expect(K.shape(x)).toEqual([3]);
  });

  it('Tensor2D', () => {
    const x = zeros([3, 2]);
    expect(K.shape(x)).toEqual([3, 2]);
  });

  it('Tensor3D', () => {
    const x = zeros([4, 3, 2]);
    expect(K.shape(x)).toEqual([4, 3, 2]);
  });

  it('Tensor4D', () => {
    const x = zeros([4, 3, 2, 1]);
    expect(K.shape(x)).toEqual([4, 3, 2, 1]);
  });
});

describe('intShape', () => {
  it('Scalar', () => {
    const x = zeros([]);
    expect(K.intShape(x)).toEqual([]);
  });

  it('Tensor1D', () => {
    const x = zeros([3]);
    expect(K.intShape(x)).toEqual([3]);
  });

  it('Tensor2D', () => {
    const x = zeros([3, 2]);
    expect(K.intShape(x)).toEqual([3, 2]);
  });

  it('Tensor3D', () => {
    const x = zeros([4, 3, 2]);
    expect(K.intShape(x)).toEqual([4, 3, 2]);
  });

  it('Tensor4D', () => {
    const x = zeros([4, 3, 2, 1]);
    expect(K.intShape(x)).toEqual([4, 3, 2, 1]);
  });
});

describe('ndim', () => {
  it('Scalar', () => {
    const x = zeros([]);
    expect(K.ndim(x)).toEqual(0);
  });

  it('Tensor1D', () => {
    const x = zeros([3]);
    expect(K.ndim(x)).toEqual(1);
  });

  it('Tensor2D', () => {
    const x = zeros([3, 2]);
    expect(K.ndim(x)).toEqual(2);
  });

  it('Tensor3D', () => {
    const x = zeros([4, 3, 2]);
    expect(K.ndim(x)).toEqual(3);
  });

  it('Tensor4D', () => {
    const x = zeros([4, 3, 2, 1]);
    expect(K.ndim(x)).toEqual(4);
  });
});

describe('dtype', () => {
  it('returns float32 for an Tensor', () => {
    const x = zeros([1]);
    expect(K.dtype(x)).toEqual('float32');
  });

  it('returns float32 for a SymbolicTensor', () => {
    const x = new SymbolicTensor('float32', [1], null, [], {});
    expect(K.dtype(x)).toEqual('float32');
  });
});


describeMathCPU('countParams', () => {
  it('Scalar', () => {
    const x = zeros([]);
    expect(K.countParams(x)).toEqual(1);
    expect(K.countParams(new LayerVariable(x).read())).toEqual(1);
  });

  it('Tensor1D', () => {
    const x = zeros([3]);
    expect(K.countParams(x)).toEqual(3);
    expect(K.countParams(new LayerVariable(x).read())).toEqual(3);
  });

  it('Tensor2D', () => {
    const x = zeros([3, 2]);
    expect(K.countParams(x)).toEqual(6);
    expect(K.countParams(new LayerVariable(x).read())).toEqual(6);
  });

  it('Tensor3D', () => {
    const x = zeros([4, 3, 2]);
    expect(K.countParams(x)).toEqual(24);
    expect(K.countParams(new LayerVariable(x).read())).toEqual(24);
  });

  it('Tensor4D', () => {
    const x = zeros([4, 3, 2, 1]);
    expect(K.countParams(x)).toEqual(24);
    expect(K.countParams(new LayerVariable(x).read())).toEqual(24);
  });
});

describeMathCPUAndGPU('cast', () => {
  it('float32 to int32', () => {
    const x =
      tensor2d([[-1.1, -1.6], [1.1, 2.2], [3.6, 4.7]], [3, 2], 'float32');
    const y = K.cast(x, 'int32');
    expect(y.dtype).toEqual('int32');
    expect(y.shape).toEqual([3, 2]);
    expect(Array.from(y.dataSync())).toEqual([-1, -1, 1, 2, 3, 4]);
  });
  it('int32 to float32', () => {
    const x = tensor2d([[-1, -1], [1, 2], [3, 4]], [3, 2], 'int32');
    const y = K.cast(x, 'float32');
    expect(y.dtype).toEqual('float32');
    expect(y.shape).toEqual([3, 2]);
    expect(Array.from(y.dataSync())).toEqual([-1, -1, 1, 2, 3, 4]);
  });
  it('float32 to bool', () => {
    const x =
      tensor2d([[-1.1, -1.6], [0.0, 2.2], [3.6, 4.7]], [3, 2], 'float32');
    const y = K.cast(x, 'bool');
    expect(y.dtype).toEqual('bool');
    expect(y.shape).toEqual([3, 2]);
    expect(Array.from(y.dataSync())).toEqual([1, 1, 0, 1, 1, 1]);
  });
  it('bool to float32', () => {
    const x = tensor2d([[0, 1], [0, 1], [1, 0]], [3, 2], 'bool');
    const y = K.cast(x, 'float32');
    expect(y.dtype).toEqual('float32');
    expect(y.shape).toEqual([3, 2]);
    expect(Array.from(y.dataSync())).toEqual([0, 1, 0, 1, 1, 0]);
  });
  it('int32 to bool', () => {
    const x = tensor2d([[-1, -2], [0, 2], [3, 4]], [3, 2], 'int32');
    const y = K.cast(x, 'bool');
    expect(y.dtype).toEqual('bool');
    expect(y.shape).toEqual([3, 2]);
    expect(Array.from(y.dataSync())).toEqual([1, 1, 0, 1, 1, 1]);
  });
  it('bool to int32', () => {
    const x = tensor2d([[0, 1], [0, 1], [1, 0]], [3, 2], 'bool');
    const y = K.cast(x, 'int32');
    expect(y.dtype).toEqual('int32');
    expect(y.shape).toEqual([3, 2]);
    expect(Array.from(y.dataSync())).toEqual([0, 1, 0, 1, 1, 0]);
  });
});

describeMathCPUAndGPU('Reshape', () => {
  it('1D', () => {
    const x = zeros([12]);
    expect(K.reshape(x, [12]).shape).toEqual([12]);
    expect(K.reshape(x, [3, 4]).shape).toEqual([3, 4]);
    expect(K.reshape(x, [2, 2, 3]).shape).toEqual([2, 2, 3]);
    expect(K.reshape(x, [1, 2, 2, 3]).shape).toEqual([1, 2, 2, 3]);
    expect(() => {
      K.reshape(x, [2, 2, 2, 3]);
    }).toThrowError();
  });

  it('Scalar', () => {
    const s = zeros([]);
    expect(K.reshape(s, []).shape).toEqual([]);
    expect(K.reshape(s, [1]).shape).toEqual([1]);
    expect(() => {
      K.reshape(s, [2]);
    }).toThrowError();
  });

  it('2D to 1D', () => {
    const x = tensor2d([[10, 20, 30], [40, 50, 60]], [2, 3]);
    const reshaped = K.reshape(x, [6]);
    expect(reshaped.shape).toEqual([6]);
    expect(reshaped.dataSync()).toEqual(new Float32Array([
      10, 20, 30, 40, 50, 60
    ]));
  });

  it('3D to 2D', () => {
    const x = tensor3d(
      [[[10, 20, 30], [40, 50, 60]], [[-10, -20, -30], [-40, -50, -60]]],
      [2, 2, 3]);
    const reshaped = K.reshape(x, [2, 6]);
    expect(reshaped.shape).toEqual([2, 6]);
    expect(reshaped.dataSync()).toEqual(new Float32Array([
      10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60
    ]));
  });
});

describeMathCPUAndGPU('expandDims', () => {
  it('Scalar to 1D', () => {
    const x = scalar(10);
    expectTensorsClose(K.expandDims(x), tensor1d([10]));
  });
  it('1D to 2D: Last dimension', () => {
    const x = tensor1d([10, 20, 30]);
    expectTensorsClose(K.expandDims(x), tensor2d([[10], [20], [30]], [3, 1]));
  });
  it('1D to 2D: First dimension', () => {
    const x = tensor1d([10, 20, 30]);
    expectTensorsClose(K.expandDims(x, 0), tensor2d([[10, 20, 30]], [1, 3]));
  });
  it('2D to 3D: Last dimension', () => {
    const x = tensor2d([[10, 20], [30, 40]], [2, 2]);
    expectTensorsClose(
      K.expandDims(x), tensor3d([[[10], [20]], [[30], [40]]], [2, 2, 1]));
  });
  it('2D to 3D: Second dimension', () => {
    const x = tensor2d([[10, 20], [30, 40]], [2, 2]);
    expectTensorsClose(
      K.expandDims(x, 1), tensor3d([[[10, 20]], [[30, 40]]], [2, 1, 2]));
  });
  it('2D to 3D: First dimension', () => {
    const x = tensor2d([[10, 20], [30, 40]], [2, 2]);
    expectTensorsClose(
      K.expandDims(x, 0), tensor3d([[[10, 20], [30, 40]]], [1, 2, 2]));
  });
});

describeMathCPUAndGPU('Repeat', () => {
  it('2D array', () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const y = K.repeat(x, 3);
    expectTensorsClose(
      y,
      tensor3d(
        [[[1, 2], [1, 2], [1, 2]], [[3, 4], [3, 4], [3, 4]]], [2, 3, 2]));
  });
  it('Non-2D array leads to AssertionError', () => {
    const x = tensor1d([1, 2, 3]);
    expect(() => K.repeat(x, 2))
      .toThrowError(
        /repeat\(\) expects a rank-2 tensor, but received a rank-1 tensor/);
  });
});

describeMathCPUAndGPU('Flatten', () => {
  it('1D Tensor', () => {
    const x = tensor1d([1, 3, 3, 7]);
    const flattend = K.flatten(x);
    expect(flattend.shape).toEqual([4]);
    expect(flattend.dataSync()).toEqual(new Float32Array([1, 3, 3, 7]));
  });

  it('2D Tensor', () => {
    const x = tensor2d([1, 3, 3, 7], [2, 2]);
    const flattend = K.flatten(x);
    expect(flattend.shape).toEqual([4]);
    expect(flattend.dataSync()).toEqual(new Float32Array([1, 3, 3, 7]));
  });

  it('3D Tensor', () => {
    const x = tensor3d(
      [[[10, 20, 30], [40, 50, 60]], [[-10, -20, -30], [-40, -50, -60]]],
      [2, 2, 3]);
    const flattend = K.flatten(x);
    expect(flattend.shape).toEqual([12]);
    expect(flattend.dataSync()).toEqual(new Float32Array([
      10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60
    ]));
  });

  it('4D Tensor', () => {
    const x = tensor4d(
      [1, 2, 3, 4, 5, 6, 7, 8, -8, -7, -6, -5, -4, -3, -2, -1], [2, 2, 2, 2]);
    const flattend = K.flatten(x);
    expect(flattend.shape).toEqual([16]);
    expect(flattend.dataSync()).toEqual(new Float32Array([
      1, 2, 3, 4, 5, 6, 7, 8, -8, -7, -6, -5, -4, -3, -2, -1
    ]));
  });
});

describeMathCPUAndGPU('batchFlatten', () => {
  it('Scalar Tensor leads to error', () => {
    const x = scalar(1337);
    expect(() => K.batchFlatten(x))
      .toThrowError(
        /batchFlatten requires a minimum rank of 2\. Got rank: 0/);
  });

  it('1D Tensor leads to error', () => {
    const x = tensor1d([1, 3, 3, 7]);
    expect(() => K.batchFlatten(x))
      .toThrowError(
        /batchFlatten requires a minimum rank of 2\. Got rank: 1/);
  });

  it('2D Tensor', () => {
    const x = tensor2d([1, 3, 3, 7], [2, 2]);
    const batchFlattened = K.batchFlatten(x);
    expect(batchFlattened.shape).toEqual([2, 2]);
    expect(batchFlattened.dataSync()).toEqual(new Float32Array([1, 3, 3, 7]));
  });

  it('3D Tensor', () => {
    const x = tensor3d(
      [[[10, 20, 30], [40, 50, 60]], [[-10, -20, -30], [-40, -50, -60]]],
      [2, 2, 3]);
    const batchFlattened = K.batchFlatten(x);
    expect(batchFlattened.shape).toEqual([2, 6]);
    expect(batchFlattened.dataSync()).toEqual(new Float32Array([
      10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60
    ]));
  });

  it('4D Tensor', () => {
    const x = tensor4d(
      [1, 2, 3, 4, 5, 6, 7, 8, -8, -7, -6, -5, -4, -3, -2, -1], [2, 2, 2, 2]);
    const batchFlattened = K.batchFlatten(x);
    expect(batchFlattened.shape).toEqual([2, 8]);
    expect(batchFlattened.dataSync()).toEqual(new Float32Array([
      1, 2, 3, 4, 5, 6, 7, 8, -8, -7, -6, -5, -4, -3, -2, -1
    ]));
  });
});

describeMathCPUAndGPU('sliceAlongFirstAxis', () => {
  const array1DData = [10, 20, 30, 40];
  it('1D', () => {
    const x = tensor1d(array1DData);
    expectTensorsClose(K.sliceAlongFirstAxis(x, 1, 2), tensor1d([20, 30]));
  });

  const array2DData = [[10, 11], [20, 21], [30, 31], [40, 41]];
  it('2D', () => {
    const x = tensor2d(array2DData, [4, 2]);
    expectTensorsClose(
      K.sliceAlongFirstAxis(x, 1, 2), tensor2d([[20, 21], [30, 31]], [2, 2]));
  });

  const array3DData = [[[10]], [[20]], [[30]], [[40]]];
  it('3D', () => {
    const x = tensor3d(array3DData, [4, 1, 1]);
    expectTensorsClose(
      K.sliceAlongFirstAxis(x, 1, 2), tensor3d([[[20]], [[30]]], [2, 1, 1]));
  });

  const array4DData = [[[[10]]], [[[20]]], [[[30]]], [[[40]]]];
  it('4D', () => {
    const x = tensor4d(array4DData, [4, 1, 1, 1]);
    expectTensorsClose(
      K.sliceAlongFirstAxis(x, 1, 2),
      tensor4d([[[[20]]], [[[30]]]], [2, 1, 1, 1]));
  });

  it('Scalar leads to error', () => {
    expect(() => {
      K.sliceAlongFirstAxis(scalar(24), 0, 1);
    }).toThrow();
  });
});

describeMathCPUAndGPU('sliceAlongLastAxis', () => {
  const array1DData = [10, 20, 30, 40];
  it('1D', () => {
    const x = tensor1d(array1DData);
    expectTensorsClose(K.sliceAlongLastAxis(x, 1, 2), tensor1d([20, 30]));
  });

  const array2DData = [[10, 11, 12, 13], [20, 21, 22, 23]];
  it('2D', () => {
    const x = tensor2d(array2DData, [2, 4]);
    expectTensorsClose(
      K.sliceAlongLastAxis(x, 1, 2), tensor2d([[11, 12], [21, 22]], [2, 2]));
  });

  const array3DData = [[[10, 20, 30, 40]]];
  it('3D', () => {
    const x = tensor3d(array3DData, [1, 1, 4]);
    expectTensorsClose(
      K.sliceAlongLastAxis(x, 1, 2), tensor3d([[[20, 30]]], [1, 1, 2]));
  });

  const array4DData = [[[[10, 20, 30, 40]]]];
  it('3D', () => {
    const x = tensor4d(array4DData, [1, 1, 1, 4]);
    expectTensorsClose(
      K.sliceAlongLastAxis(x, 1, 2), tensor4d([[[[20, 30]]]], [1, 1, 1, 2]));
  });
});


describeMathCPUAndGPU('sliceAlongAxis', () => {
  it('1D', () => {
    const array1DData = [10, 20, 30, 40];
    const x = tensor1d(array1DData);
    expectTensorsClose(K.sliceAlongAxis(x, 1, 2, 1), tensor1d([20, 30]));
  });

  const array2DData = [[10, 11], [20, 21], [30, 31], [40, 41]];
  it('2D-1', () => {
    const x = tensor2d(array2DData, [4, 2]);
    expectTensorsClose(
      K.sliceAlongAxis(x, 1, 2, 1), tensor2d([[20, 21], [30, 31]], [2, 2]));
  });

  it('2D-2', () => {
    const x = tensor2d(array2DData, [4, 2]);
    expectTensorsClose(
      K.sliceAlongAxis(x, 0, 1, 2),
      tensor2d([[10], [20], [30], [40]], [4, 1]));
  });

  const array3DData = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
  it('3D-1', () => {
    const x = tensor3d(array3DData, [2, 2, 2]);
    expectTensorsClose(
      K.sliceAlongAxis(x, 0, 1, 1), tensor3d([[[1, 2], [3, 4]]], [1, 2, 2]));
  });
  it('3D-2', () => {
    const x = tensor3d(array3DData, [2, 2, 2]);
    expectTensorsClose(
      K.sliceAlongAxis(x, 0, 1, 2),
      tensor3d([[[1, 2]], [[5, 6]]], [2, 1, 2]));
  });
  it('3D-3', () => {
    const x = tensor3d(array3DData, [2, 2, 2]);
    expectTensorsClose(
      K.sliceAlongAxis(x, 0, 1, 3),
      tensor3d([[[1], [3]], [[5], [7]]], [2, 2, 1]));
  });


  it('4D', () => {
    const array4DData = [[[[10, 1]]], [[[20, 2]]], [[[30, 3]]], [[[40, 4]]]];
    const x = tensor4d(array4DData, [4, 1, 1, 2]);
    expectTensorsClose(
      K.sliceAlongAxis(x, 0, 1, 4),
      tensor4d([[[[10]]], [[[20]]], [[[30]]], [[[40]]]], [4, 1, 1, 1]));
  });
});
describeMathCPUAndGPU('normalizeBatchInTraining', () => {
  // The reference values for assertion below can be obtained with Python code
  // as the following:
  // ```python
  // import keras
  // import numpy as np
  // import tensorflow as tf
  //
  // with tf.Session() as sess:
  //   x = tf.Variable(np.array(
  //       [[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]], dtype=np.float32))
  //   gamma = tf.Variable(np.array([1, 1, 1, 1], dtype=np.float32))
  //   beta = tf.Variable(np.array([0, 0, 0, 0], dtype=np.float32))
  //   reduction_axes = [0]
  //   normed, mean, variance = keras.backend.normalize_batch_in_training(
  //       x, gamma, beta, reduction_axes)
  //   print(normed)
  //   print(mean)
  //   print(variance)
  // ```

  it('2D, no broadcasting', () => {
    const x = tensor2d([[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]], [3, 4]);
    const gamma = tensor1d([1, 1, 1, 1]);
    const beta = tensor1d([0, 0, 0, 0]);
    const reductionAxes = [0];
    const [normed, mean, variance] =
      K.normalizeBatchInTraining(x, gamma, beta, reductionAxes);
    expectTensorsClose(
      normed,
      tensor2d(
        [
          [-0.805371, -0.9502233, -1.1624058, -1.3885813],
          [-0.6040282, -0.4319197, -0.11624074, 0.46286058],
          [1.4093992, 1.3821429, 1.2786462, 0.92572117]
        ],
        [3, 4]));
    expectTensorsClose(mean, tensor1d([5.0, 5.6666665, 6.3333335, 7.0]));
    expectTensorsClose(
      variance, tensor1d([24.666666, 14.888889, 8.222222, 4.6666665]));
  });

  it('3D, no broadcasting', () => {
    const x = tensor3d(
      [[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]], [3, 2, 2]);
    const gamma = tensor1d([1, 1]);
    const beta = tensor1d([0, 0]);
    const reductionAxes = [0, 1];
    const [normed, mean, variance] =
      K.normalizeBatchInTraining(x, gamma, beta, reductionAxes);
    expectTensorsClose(
      normed,
      tensor3d(
        [
          [[-1.1355163, -1.3552775], [-0.6488664, -0.7297648]],
          [[-0.8921913, -0.7297648], [0.08110833, 0.5212605]],
          [[1.5410578, 1.4595294], [1.0544081, 0.8340168]]
        ],
        [3, 2, 2]));
    expectTensorsClose(mean, tensor1d([5.6666665, 6.3333335]));
    expectTensorsClose(variance, tensor1d([16.88889, 10.222222]));
  });

  it('3D, broadcasting', () => {
    const x = tensor3d(
      [[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]], [3, 2, 2]);
    const gamma = tensor2d([[1, 1], [1, 1]], [2, 2]);
    const beta = tensor2d([[0, 0], [0, 0]], [2, 2]);
    const reductionAxes = [0];
    const [normed, mean, variance] =
      K.normalizeBatchInTraining(x, gamma, beta, reductionAxes);
    expectTensorsClose(
      normed,
      tensor3d(
        [
          [[-0.805371, -0.9502233], [-1.1624058, -1.3885813]],
          [[-0.6040282, -0.4319197], [-0.11624074, 0.46286058]],
          [[1.4093992, 1.3821429], [1.2786462, 0.92572117]]
        ],
        [3, 2, 2]));
    expectTensorsClose(
      mean, tensor2d([[5, 5.6666665], [6.3333335, 7]], [2, 2]));
    expectTensorsClose(
      variance,
      tensor2d([[24.666666, 14.888889], [8.222222, 4.6666665]], [2, 2]));
  });

  it('4D, broadcasting', () => {
    const x = tensor4d(
      [[[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]]],
      [1, 3, 2, 2]);
    const gamma = tensor2d([[1, 1], [1, 1]], [2, 2]);
    const beta = tensor2d([[0, 0], [0, 0]], [2, 2]);
    const reductionAxes = [0, 1];
    const [normed, mean, variance] =
      K.normalizeBatchInTraining(x, gamma, beta, reductionAxes);
    expectTensorsClose(
      normed,
      tensor4d(
        [[
          [[-0.805371, -0.9502233], [-1.1624058, -1.3885813]],
          [[-0.6040282, -0.4319197], [-0.11624074, 0.46286058]],
          [[1.4093992, 1.3821429], [1.2786462, 0.92572117]]
        ]],
        [1, 3, 2, 2]));
    expectTensorsClose(
      mean, tensor2d([[5, 5.6666665], [6.3333335, 7]], [2, 2]));
    expectTensorsClose(
      variance,
      tensor2d([[24.666666, 14.888889], [8.222222, 4.6666665]], [2, 2]));
  });
});

describeMathCPUAndGPU('concatenate', () => {
  it('1D', () => {
    const x = tensor1d([1, 2, 3, 4]);
    const y = tensor1d([-1, -2, -3, -4]);
    const expected = tensor1d([1, 2, 3, 4, -1, -2, -3, -4]);
    expectTensorsClose(K.concatenate([x, y]), expected);
    expectTensorsClose(K.concatenate([x, y], -1), expected);
    expectTensorsClose(K.concatenate([x, y], 0), expected);
  });
  it('2D', () => {
    const x = tensor2d([1, 2, 3, 4], [2, 2]);
    const y = tensor2d([-1, -2, -3, -4], [2, 2]);
    let expected = tensor2d([1, 2, -1, -2, 3, 4, -3, -4], [2, 4]);
    expectTensorsClose(K.concatenate([x, y]), expected);
    expectTensorsClose(K.concatenate([x, y], -1), expected);
    expectTensorsClose(K.concatenate([x, y], 1), expected);
    expected = tensor2d([1, 2, 3, 4, -1, -2, -3, -4], [4, 2]);
    expectTensorsClose(K.concatenate([x, y], 0), expected);
  });
  it('3D', () => {
    const x = tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const y = tensor3d([-1, -2, -3, -4], [2, 2, 1]);
    let expected = tensor3d([1, -1, 2, -2, 3, -3, 4, -4], [2, 2, 2]);
    expectTensorsClose(K.concatenate([x, y]), expected);
    expectTensorsClose(K.concatenate([x, y], -1), expected);
    expectTensorsClose(K.concatenate([x, y], 2), expected);
    expected = tensor3d([1, 2, -1, -2, 3, 4, -3, -4], [2, 4, 1]);
    expectTensorsClose(K.concatenate([x, y], 1), expected);
    expected = tensor3d([1, 2, 3, 4, -1, -2, -3, -4], [4, 2, 1]);
    expectTensorsClose(K.concatenate([x, y], 0), expected);
  });
  it('3D', () => {
    const x = tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const y = tensor4d([-1, -2, -3, -4, -5, -6, -7, -8], [2, 2, 2, 1]);
    let expected = tensor4d(
      [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8], [2, 2, 2, 2]);
    expectTensorsClose(K.concatenate([x, y]), expected);
    expectTensorsClose(K.concatenate([x, y], -1), expected);
    expectTensorsClose(K.concatenate([x, y], 3), expected);
    expected = tensor4d(
      [1, 2, -1, -2, 3, 4, -3, -4, 5, 6, -5, -6, 7, 8, -7, -8], [2, 2, 4, 1]);
    expectTensorsClose(K.concatenate([x, y], 2), expected);
    expected = tensor4d(
      [1, 2, 3, 4, -1, -2, -3, -4, 5, 6, 7, 8, -5, -6, -7, -8], [2, 4, 2, 1]);
    expectTensorsClose(K.concatenate([x, y], 1), expected);
    expected = tensor4d(
      [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8], [4, 2, 2, 1]);
    expectTensorsClose(K.concatenate([x, y], 0), expected);
  });
});

describeMathCPUAndGPU('concatAlongFirstAxis', () => {
  const array1DData1 = [10, 20, 30, 40];
  const array1DData2 = [-10, -20, -30, -40];
  it('1D', () => {
    const a = tensor1d(array1DData1);
    const b = tensor1d(array1DData2);
    expectTensorsClose(
      K.concatAlongFirstAxis(a, b),
      tensor1d([10, 20, 30, 40, -10, -20, -30, -40]));
  });

  const array2DData1 = [[10, 11], [20, 21]];
  const array2DData2 = [[30, 31], [40, 41]];
  it('2D', () => {
    const a = tensor2d(array2DData1, [2, 2]);
    const b = tensor2d(array2DData2, [2, 2]);
    expectTensorsClose(
      K.concatAlongFirstAxis(a, b),
      tensor2d([[10, 11], [20, 21], [30, 31], [40, 41]], [4, 2]));
  });

  const array3DData1 = [[[10]], [[20]]];
  const array3DData2 = [[[30]], [[40]]];
  it('3D', () => {
    const a = tensor3d(array3DData1, [2, 1, 1]);
    const b = tensor3d(array3DData2, [2, 1, 1]);
    expectTensorsClose(
      K.concatAlongFirstAxis(a, b),
      tensor3d([[[10]], [[20]], [[30]], [[40]]], [4, 1, 1]));
  });

  const array4DData1 = [[[[10]]], [[[20]]]];
  const array4DData2 = [[[[30]]], [[[40]]]];
  it('4D', () => {
    const a = tensor4d(array4DData1, [2, 1, 1, 1]);
    const b = tensor4d(array4DData2, [2, 1, 1, 1]);
    expectTensorsClose(
      K.concatAlongFirstAxis(a, b),
      tensor4d([[[[10]]], [[[20]]], [[[30]]], [[[40]]]], [4, 1, 1, 1]));
  });

  it('Scalar leads to error', () => {
    expect(() => {
      K.concatAlongFirstAxis(scalar(24), scalar(-24));
    }).toThrow();
  });
});

describeMathCPUAndGPU('tile', () => {
  it('1D, n is number', () => {
    const x = tensor1d([1, 3, 3, 7]);
    const n = 3;
    const y = K.tile(x, n);
    expectTensorsClose(y, tensor1d([1, 3, 3, 7, 1, 3, 3, 7, 1, 3, 3, 7]));
  });
  it('1D, n is number Array', () => {
    const x = tensor1d([1, 3, 3, 7]);
    const n = [3];
    const y = K.tile(x, n);
    expectTensorsClose(y, tensor1d([1, 3, 3, 7, 1, 3, 3, 7, 1, 3, 3, 7]));
  });
  it('2D', () => {
    const x = tensor2d([[1, 3], [3, 7]], [2, 2]);
    const n = [2, 3];
    const y = K.tile(x, n);
    expectTensorsClose(
      y,
      tensor2d(
        [
          [1, 3, 1, 3, 1, 3], [3, 7, 3, 7, 3, 7], [1, 3, 1, 3, 1, 3],
          [3, 7, 3, 7, 3, 7]
        ],
        [4, 6]));
  });
  it('3D', () => {
    const x = tensor3d([[[1]]], [1, 1, 1]);
    const n = [2, 3, 4];
    const y = K.tile(x, n);
    expectTensorsClose(y, tfc.ones([2, 3, 4]));
  });
  it('Mismatch in x dimensions and n length leads to exception', () => {
    expect(() => K.tile(tfc.zeros([2, 2]), 1))
      .toThrowError(/The length of input n \(1\) does not match .*2/);
  });
});

describeMathCPUAndGPU('Identity', () => {
  it('Scalar', () => {
    const s = scalar(12);
    const sIdentity = K.identity(s);
    expect(sIdentity.shape).toEqual([]);
    expect(sIdentity.dataSync()).toEqual(new Float32Array([12]));
  });

  it('1D', () => {
    const v = tensor1d([-12, 12]);
    const vIdentity = K.identity(v);
    expect(vIdentity.shape).toEqual([2]);
    expect(vIdentity.dataSync()).toEqual(new Float32Array([-12, 12]));
  });

  it('2D', () => {
    const m = tensor2d([[-12, 12], [-10, 10]], [2, 2]);
    const mIdentity = K.identity(m);
    expect(mIdentity.shape).toEqual([2, 2]);
    expect(mIdentity.dataSync()).toEqual(new Float32Array([-12, 12, -10, 10]));
  });
});

describeMathCPUAndGPU('scalarTimesArray', () => {
  it('Scalar x Scalar', () => {
    expectTensorsClose(K.scalarTimesArray(scalar(-2), scalar(-3)), scalar(6));
  });
  it('Scalar x 4D', () => {
    const y = K.scalarTimesArray(scalar(-2), tfc.ones([2, 2, 2, 2]));
    expect(y.shape).toEqual([2, 2, 2, 2]);
    const yValues = Array.from(y.dataSync());
    expect(unique(yValues)).toEqual([-2]);
  });
});

describeMathCPUAndGPU('scalarPlusArray', () => {
  it('Scalar + Scalar', () => {
    expectTensorsClose(K.scalarPlusArray(scalar(-2), scalar(-3)), scalar(-5));
  });
  it('Scalar + 4D', () => {
    const shape = [2, 2, 2, 2];
    const y = K.scalarPlusArray(scalar(-1), tfc.ones(shape));
    expectTensorsClose(y, tfc.zeros(shape));
  });
});


describeMathCPUAndGPU('randomNormal', () => {
  const dtypes:DataType[] = ['float32', 'int32'];
  for (const dtype  of dtypes) {
    // TODO(bileschi): Add probabilistic assertions on values here.
    it(`Scalar ${dtype}`, () => {
      const s = K.randomNormal([], 0, 10, dtype as 'float32'|'int32');
      expect(K.shape(s)).toEqual([]);
    });

    it(`1D ${dtype}`, () => {
      const v = K.randomNormal([20], 0, 2, dtype as 'float32'|'int32');
      expect(K.shape(v)).toEqual([20]);
    });

    it(`2D ${dtype}`, () => {
      const x = K.randomNormal([3, 20], -10, 20, dtype as 'float32'|'int32');
      expect(K.shape(x)).toEqual([3, 20]);
    });

    it(`3D ${dtype}`, () => {
      const y = K.randomNormal([2, 3, 4], 100, 10, dtype as 'float32'|'int32');
      expect(K.shape(y)).toEqual([2, 3, 4]);
    });
  }
});

describeMathCPUAndGPU('dot', () => {
  it('2D x 2D', () => {
    const x = tensor2d([[1, 0], [0, -1]], [2, 2]);
    const y = tensor2d([[3], [4]], [2, 1]);
    const output = K.dot(x, y);
    expectTensorsClose(output, tensor2d([[3], [-4]], [2, 1]));
  });
  it('2D x 2D: Incompatible dimensions', () => {
    const x = tensor2d([[1, 0], [0, -1]], [2, 2]);
    const y = tensor2d([[3], [4], [5]], [3, 1]);
    expect(() => K.dot(x, y)).toThrowError();
  });
  it('3D x 2D', () => {
    const x = tensor3d([[[1, 0], [0, -1]], [[-2, 0], [0, -2]]], [2, 2, 2]);
    const y = tensor2d([[-1], [1]], [2, 1]);
    expectTensorsClose(
      K.dot(x, y), tensor3d([[[-1], [-1]], [[2], [-2]]], [2, 2, 1]));
  });
  it('2D x 1D leads to error', () => {
    const x = tensor2d([[1, 0], [0, -1]], [2, 2]);
    const y = tensor1d([3, 4]);
    expect(() => K.dot(x, y)).toThrowError();
  });
  it('2D x Scalar leads to error', () => {
    const x = tensor2d([[1]], [1, 1]);
    const y = scalar(10);
    expect(() => K.dot(x, y)).toThrowError();
  });
  it('1D x 1D leads to error', () => {
    const x = tensor1d([1, 2]);
    const y = tensor1d([3, 4]);
    expect(() => K.dot(x, y)).toThrowError();
  });
});

describeMathCPUAndGPU('sign', () => {
  it('Scalar', () => {
    expectTensorsClose(K.sign(scalar(0)), scalar(0));
    expectTensorsClose(K.sign(scalar(0.5)), scalar(1));
    expectTensorsClose(K.sign(scalar(-0.5)), scalar(-1));
  });
  it('1D', () => {
    const x = tensor1d([1, 2, -1, 0, 3, -4]);
    expectTensorsClose(K.sign(x), tensor1d([1, 1, -1, 0, 1, -1]));
  });
  it('2D', () => {
    const x = tensor2d([[1, 2, -1], [0, 3, -4]], [2, 3]);
    expectTensorsClose(K.sign(x), tensor2d([[1, 1, -1], [0, 1, -1]], [2, 3]));
  });
});


describeMathCPUAndGPU('qr', () => {
  it('1x1', () => {
    const x = tensor2d([[10]], [1, 1]);
    const [q, r] = K.qr(x);
    expectTensorsClose(q, tensor2d([[-1]], [1, 1]));
    expectTensorsClose(r, tensor2d([[-10]], [1, 1]));
  });

  it('2x2', () => {
    const x = tensor2d([[1, 3], [-2, -4]], [2, 2]);
    const [q, r] = K.qr(x);
    expectTensorsClose(
      q, tensor2d([[-0.4472, -0.8944], [0.8944, -0.4472]], [2, 2]));
    expectTensorsClose(r, tensor2d([[-2.2361, -4.9193], [0, -0.8944]], [2, 2]));
  });

  it('3x3', () => {
    const x = tensor2d([[1, 3, 2], [-2, 0, 7], [8, -9, 4]], [3, 3]);
    const [q, r] = K.qr(x);
    expectTensorsClose(
      q,
      tensor2d(
        [
          [-0.1204, 0.8729, 0.4729], [0.2408, -0.4364, 0.8669],
          [-0.9631, -0.2182, 0.1576]
        ],
        [3, 3]));
    expectTensorsClose(
      r,
      tensor2d(
        [[-8.3066, 8.3066, -2.4077], [0, 4.5826, -2.1822], [0, 0, 7.6447]],
        [3, 3]));
  });

  it('3x2', () => {
    const x = tensor2d([[1, 2], [3, -3], [-2, 1]], [3, 2]);
    const [q, r] = K.qr(x);
    expectTensorsClose(
      q,
      tensor2d(
        [
          [-0.2673, 0.9221, 0.2798], [-0.8018, -0.3738, 0.4663],
          [0.5345, -0.0997, 0.8393]
        ],
        [3, 3]));
    expectTensorsClose(
      r, tensor2d([[-3.7417, 2.4054], [0, 2.8661], [0, 0]], [3, 2]));
  });

  it('does not leak memory', () => {
    const x = tensor2d([[1, 3], [-2, -4]], [2, 2]);
    // The first call to qr creates and keeps internal singleton tensors.
    // Subsequent calls should always create exactly two tensors.
    K.qr(x);
    // Count before real call.
    const numTensors = memory().numTensors;
    K.qr(x);
    expect(memory().numTensors).toEqual(numTensors + 2);
  });

  it('Incorrect shape leads to error', () => {
    const x = tensor2d([[1, 2, 3], [-3, -2, 1]], [2, 3]);
    expect(() => K.qr(x)).toThrowError(/requires.*shape/);
  });
});

describeMathCPUAndGPU('OneHot', () => {
  it('Unsupported indices', () => {
    const numClasses = 2;
    const indices = tensor2d([[-12, 12], [-10, 10]], [2, 2]);
    expect(() => {
      K.oneHot(indices, numClasses);
    }).toThrowError();
  });
  it('Unsupported numClasses', () => {
    const numClasses = 1;
    const indices = tensor1d([2, 2]);
    expect(() => {
      K.oneHot(indices, numClasses);
    }).toThrowError();
  });
  it('Supported use case', () => {
    const numClasses = 5;
    const indices = tensor1d([1, 3]);
    expectTensorsClose(
      K.oneHot(indices, numClasses),
      tensor2d([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]], [2, 5]));
  });
});

describeMathCPUAndGPU('Gather', () => {
  it('1D, Array of numbers with repeats', () => {
    expectTensorsClose(
      K.gather(tensor1d([0, 10, 20, 30]), [2, 2, 3, 1]),
      tensor1d([20, 20, 30, 10]));
  });
  it('2D, Array of numbers', () => {
    expectTensorsClose(
      K.gather(tensor2d([[10, 20], [30, 40], [50, 60]], [3, 2]), [2, 0]),
      tensor2d([[50, 60], [10, 20]], [2, 2]));
  });
  it('2D, Tensor1D', () => {
    expectTensorsClose(
      K.gather(
        tensor2d([[10, 20], [30, 40], [50, 60]], [3, 2]), tensor1d([2, 1])),
      tensor2d([[50, 60], [30, 40]], [2, 2]));
  });
  it('3D, Tensor1D', () => {
    expectTensorsClose(
      K.gather(
        tensor3d([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], [2, 2, 2]),
        tensor1d([1, 0])),
      tensor3d([[[50, 60], [70, 80]], [[10, 20], [30, 40]]], [2, 2, 2]));
  });
  it('2D, Non-default axis', () => {
    expectTensorsClose(
      K.gather(tensor2d([[10, 20], [30, 40], [50, 60]], [3, 2]), [1], 1),
      tensor2d([[20], [40], [60]], [3, 1]));
  });
});


describeMathCPUAndGPU('Square', () => {
  it('Element-wise square', () => {
    expectTensorsClose(
      K.square(tensor2d([[1, -2], [-3, 4]], [2, 2])),
      tensor2d([1, 4, 9, 16], [2, 2]));
  });
});

describeMathCPUAndGPU('Pow', () => {
  it('Element-wise Pow: Positive Scalar', () => {
    expectTensorsClose(
      K.pow(tensor2d([[1, 1.5], [2, 2.5]], [2, 2]), scalar(2, 'int32')),
      tensor2d([[1, 2.25], [4, 6.25]], [2, 2]));
  });
  it('Element-wise Pow: Negative Scalar', () => {
    expectTensorsClose(
      K.pow(tensor2d([[1, 1.5], [2, 2.5]], [2, 2]), scalar(-2, 'int32')),
      tensor2d(
        [[1, 1 / (1.5 * 1.5)], [1 / (2 * 2), 1 / (2.5 * 2.5)]], [2, 2]));
  });
  it('Element-wise Pow: Zero Scalar', () => {
    expectTensorsClose(
      K.pow(tensor2d([[1, 1.5], [2, 2.5]], [2, 2]), scalar(0, 'int32')),
      tensor2d([[1, 1], [1, 1]], [2, 2]));
  });
  it('Element-wise Pow: number', () => {
    expectTensorsClose(
      K.pow(tensor2d([[1, 1.5], [2, 2.5]], [2, 2]), 2),
      tensor2d([[1, 2.25], [4, 6.25]], [2, 2]));
  });
});

describeMathCPUAndGPU('softsign', () => {
  it('Element-wise softsign', () => {
    expectTensorsClose(
      tfc.tanh(tensor2d([[-2, -1], [1, 2]], [2, 2])),
      tensor2d(
        [Math.tanh(-2), Math.tanh(-1), Math.tanh(1), Math.tanh(2)],
        [2, 2]));
  });
});


describeMathCPUAndGPU('batchNormalization', () => {
  it('2D, no broadcast, no gamma, no beta', () => {
    const x = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const mean = tensor2d([[5, 5], [5, 5]], [2, 2]);
    const variance = tensor2d([[4, 16], [4, 16]], [2, 2]);
    expectTensorsClose(
      K.batchNormalization(x, mean, variance, null, null, 0),
      tensor2d([[2.5, 3.75], [12.5, 8.75]], [2, 2]));
  });
  it('2D, no broadcast, no gamma, no beta, custom epsilon', () => {
    const x = tensor2d([[30, 30], [60, 60]], [2, 2]);
    const mean = tensor2d([[0, 0], [0, 0]], [2, 2]);
    const variance = tensor2d([[7, 7], [7, 7]], [2, 2]);
    expectTensorsClose(
      K.batchNormalization(x, mean, variance, null, null, 2),
      tensor2d([[10, 10], [20, 20]], [2, 2]));
  });
  it('2D, no broadcast, gamma, no beta', () => {
    const x = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const mean = tensor2d([[5, 5], [5, 5]], [2, 2]);
    const variance = tensor2d([[4, 16], [4, 16]], [2, 2]);
    const gamma = tensor2d([[1, 2], [3, 4]], [2, 2]);
    expectTensorsClose(
      K.batchNormalization(x, mean, variance, null, gamma, 0),
      tensor2d([[2.5, 7.5], [37.5, 35]], [2, 2]));
  });
  it('2D, no broadcast, gamma, beta', () => {
    const x = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const mean = tensor2d([[5, 5], [5, 5]], [2, 2]);
    const variance = tensor2d([[4, 16], [4, 16]], [2, 2]);
    const gamma = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const beta = tensor2d([[-1, -1], [-2, -2]], [2, 2]);
    expectTensorsClose(
      K.batchNormalization(x, mean, variance, beta, gamma, 0),
      tensor2d([[1.5, 6.5], [35.5, 33]], [2, 2]));
  });
  it('2D, broadcast, gamma, beta', () => {
    const x = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const mean = tensor1d([2, 5]);
    const variance = tensor1d([1, 4]);
    const gamma = tensor1d([3, 4]);
    const beta = tensor1d([-1, -2]);
    expectTensorsClose(
      K.batchNormalization(x, mean, variance, beta, gamma, 0),
      tensor2d([[23, 28], [83, 68]], [2, 2]));
  });
  it('3D, no broadcast, no gamma, no beta', () => {
    const x = tensor3d([[[10, 20], [30, 40]], [[10, 20], [30, 40]]], [2, 2, 2]);
    const mean = tensor3d([[[5, 5], [5, 5]], [[5, 5], [5, 5]]], [2, 2, 2]);
    const variance =
      tensor3d([[[4, 16], [4, 16]], [[16, 25], [16, 25]]], [2, 2, 2]);
    expectTensorsClose(
      K.batchNormalization(x, mean, variance, null, null, 0),
      tensor3d(
        [[[2.5, 3.75], [12.5, 8.75]], [[1.25, 3], [6.25, 7]]], [2, 2, 2]));
  });
  it('3D, no broadcast, gamma, beta', () => {
    const x = tensor3d([[[10, 20], [30, 40]], [[10, 20], [30, 40]]], [2, 2, 2]);
    const mean = tensor3d([[[5, 5], [5, 5]], [[5, 5], [5, 5]]], [2, 2, 2]);
    const variance =
      tensor3d([[[4, 16], [4, 16]], [[16, 25], [16, 25]]], [2, 2, 2]);
    const gamma = tensor3d([[[2, 2], [2, 2]], [[4, 4], [4, 4]]], [2, 2, 2]);
    const beta =
      tensor3d([[[-1, -1], [-2, -2]], [[-1, -1], [-2, -2]]], [2, 2, 2]);
    expectTensorsClose(
      K.batchNormalization(x, mean, variance, beta, gamma, 0),
      tensor3d([[[4, 6.5], [23, 15.5]], [[4, 11], [23, 26]]], [2, 2, 2]));
  });
  it('3D, broadcast, gamma, beta', () => {
    const x = tensor3d([[[10, 20], [30, 40]], [[10, 20], [30, 40]]], [2, 2, 2]);
    const mean = tensor1d([5, 5]);
    const variance = tensor1d([4, 16]);
    const gamma = tensor1d([2, 4]);
    const beta = tensor1d([-1, -2]);
    expectTensorsClose(
      K.batchNormalization(x, mean, variance, beta, gamma, 0),
      tensor3d([[[4, 13], [24, 33]], [[4, 13], [24, 33]]], [2, 2, 2]));
  });
  it('4D, no broadcast, no gamma, no beta', () => {
    const x = tensor4d(
      [
        [[[10, 20], [30, 40]], [[10, 20], [30, 40]]],
        [[[-10, -20], [-30, -40]], [[-10, -20], [-30, -40]]]
      ],
      [2, 2, 2, 2]);
    const mean = tensor4d(
      [
        [[[5, 5], [5, 5]], [[5, 5], [5, 5]]],
        [[[-5, -5], [-5, -5]], [[-5, -5], [-5, -5]]]
      ],
      [2, 2, 2, 2]);
    const variance = tensor4d(
      [
        [[[4, 16], [4, 16]], [[16, 25], [16, 25]]],
        [[[4, 16], [4, 16]], [[16, 25], [16, 25]]]
      ],
      [2, 2, 2, 2]);
    expectTensorsClose(
      K.batchNormalization(x, mean, variance, null, null, 0),
      tensor4d(
        [
          [[[2.5, 3.75], [12.5, 8.75]], [[1.25, 3], [6.25, 7]]],
          [[[-2.5, -3.75], [-12.5, -8.75]], [[-1.25, -3], [-6.25, -7]]]
        ],
        [2, 2, 2, 2]));
  });
  it('4D, no broadcast, gamma, beta', () => {
    const x = tensor4d(
      [
        [[[10, 20], [30, 40]], [[10, 20], [30, 40]]],
        [[[-10, -20], [-30, -40]], [[-10, -20], [-30, -40]]]
      ],
      [2, 2, 2, 2]);
    const mean = tensor4d(
      [
        [[[5, 5], [5, 5]], [[5, 5], [5, 5]]],
        [[[-5, -5], [-5, -5]], [[-5, -5], [-5, -5]]]
      ],
      [2, 2, 2, 2]);
    const variance = tensor4d(
      [
        [[[4, 16], [4, 16]], [[16, 25], [16, 25]]],
        [[[4, 16], [4, 16]], [[16, 25], [16, 25]]]
      ],
      [2, 2, 2, 2]);
    const gamma = tensor4d(
      [
        [[[2, 2], [2, 2]], [[4, 4], [4, 4]]],
        [[[2, 2], [2, 2]], [[4, 4], [4, 4]]]
      ],
      [2, 2, 2, 2]);
    const beta = tensor4d(
      [
        [[[-1, -1], [-2, -2]], [[-1, -1], [-2, -2]]],
        [[[1, 1], [2, 2]], [[1, 1], [2, 2]]]
      ],
      [2, 2, 2, 2]);
    expectTensorsClose(
      K.batchNormalization(x, mean, variance, beta, gamma, 0),
      tensor4d(
        [
          [[[4, 6.5], [23, 15.5]], [[4, 11], [23, 26]]],
          [[[-4, -6.5], [-23, -15.5]], [[-4, -11], [-23, -26]]]
        ],
        [2, 2, 2, 2]));
  });
  it('4D, broadcast, gamma, beta', () => {
    const x = tensor4d(
      [[[[10, 20], [30, 40]]], [[[10, 20], [30, 40]]]], [2, 1, 2, 2]);
    const mean = tensor1d([5, 5]);
    const variance = tensor1d([4, 16]);
    const gamma = tensor1d([2, 4]);
    const beta = tensor1d([-1, -2]);
    expectTensorsClose(
      K.batchNormalization(x, mean, variance, beta, gamma, 0),
      tensor4d([[[[4, 13], [24, 33]]], [[[4, 13], [24, 33]]]], [2, 1, 2, 2]));
  });
});

describeMathCPUAndGPU('dropout', () => {
  const dropoutLevels = [0, 0.75];
  for (const dropoutLevel of dropoutLevels) {
    it(`Level = ${dropoutLevel}`, () => {
      const x = tensor2d(range(1, 21), [10, 2]);
      const y = K.dropout(x, scalar(dropoutLevel));
      expect(y.dtype).toEqual(x.dtype);
      expect(y.shape).toEqual(x.shape);
      const xValue = x.dataSync();
      const yValue = y.dataSync();
      let nKept = 0;
      for (let i = 0; i < xValue.length; ++i) {
        if (yValue[i] !== 0) {
          nKept++;
          expect(yValue[i]).toBeCloseTo(1 / (1 - dropoutLevel) * xValue[i]);
        }
      }
      const numel = K.countParams(x);
      if (dropoutLevel === 0) {
        expect(nKept).toEqual(numel);
      } else {
        expect(nKept).toBeLessThan(numel);
      }
    });
  }
});

describeMathCPUAndGPU('l2Normalize', () => {
  it('normalizes with no axis defined.', () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const norm = Math.sqrt(1 * 1 + 2 * 2 + 3 * 3 + 4 * 4);
    const expected =
      tensor2d([[1 / norm, 2 / norm], [3 / norm, 4 / norm]], [2, 2]);
    const result = K.l2Normalize(x);
    expectTensorsClose(result, expected);
  });

  it('normalizes along axis = -1.', () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const firstNorm = Math.sqrt(1 * 1 + 2 * 2);
    const secondNorm = Math.sqrt(3 * 3 + 4 * 4);
    const expected = tensor2d(
      [[1 / firstNorm, 2 / firstNorm], [3 / secondNorm, 4 / secondNorm]],
      [2, 2]);
    const result = K.l2Normalize(x, -1);
    expectTensorsClose(result, expected);
  });

  it('normalizes with zeros.', () => {
    const x = zeros([2, 2]);
    const result = K.l2Normalize(x);
    expectTensorsClose(result, x);
  });
});


describeMathCPUAndGPU('biasAdd', () => {
  it('1D + 1D', () => {
    const x = tfc.ones([2]);
    const y = tensor1d([-1, 1]);
    expectTensorsClose(K.biasAdd(x, y), tensor1d([0, 2]));
  });
  it('2D + 1D', () => {
    const x = tfc.ones([2, 2]);
    const y = tensor1d([-1, 1]);
    expectTensorsClose(K.biasAdd(x, y), tensor2d([[0, 2], [0, 2]], [2, 2]));
  });
  it('3D + 1D', () => {
    const x = tfc.ones([2, 2, 2]);
    const y = tensor1d([-1, 1]);
    expectTensorsClose(
      K.biasAdd(x, y),
      tensor3d([[[0, 2], [0, 2]], [[0, 2], [0, 2]]], [2, 2, 2]));
  });
  it('4D + 1D', () => {
    const x = tfc.ones([1, 2, 2, 2]);
    const y = tensor1d([-1, 1]);
    expectTensorsClose(
      K.biasAdd(x, y),
      tensor4d([[[[0, 2], [0, 2]], [[0, 2], [0, 2]]]], [1, 2, 2, 2]));
  });
  it('2D + 1D: Incompatible size', () => {
    const x = tfc.ones([2, 2]);
    const y = tensor1d([-1, 0, 1]);
    expect(() => K.biasAdd(x, y)).toThrowError();
  });
  it('3D + 2D leads to error', () => {
    const x = tfc.ones([2, 2, 2]);
    const y = tfc.ones([2, 2]);
    expect(() => K.biasAdd(x, y)).toThrowError();
  });
});

describeMathCPUAndGPU('elu', () => {
  it('elu', () => {
    const xData = [-1, 0, 1, -1];
    expectTensorsClose(
      K.elu(tensor2d(xData, [2, 2])),
      tensor2d(xData.map(x => x < 0 ? Math.exp(x) - 1 : x), [2, 2]));
  });
});

describeMathCPUAndGPU('softsign', () => {
  it('softsign', () => {
    const xData = [-1, 0, 1, -1];
    expectTensorsClose(
      K.softsign(tensor2d(xData, [2, 2])),
      tensor2d(xData.map(x => x / (Math.abs(x) + 1)), [2, 2]));
  });
  it ('Does not leak', () => {
    const input = tensor2d([-1, 0, 1, -1], [2, 2]);
    expectNoLeakedTensors(() => K.softsign(input), 1);
  });
});

describe('floatx ', () => {
  it('returns "float32"', () => {
    expect(K.floatx()).toEqual('float32');
  });
});

describe('Name scope ', () => {
  it('returns function\'s value from the name scope.', () => {
    const name = 'name';
    const val = 'val';
    const fn = () => val;
    expect(K.nameScope<string>(name, fn)).toEqual(val);
  });

  it('re-throws exception.', () => {
    const exceptionValue = 'exception';
    const exceptionFn = () => {
      throw new Error(exceptionValue);
    };
    const nameScopeFn = () => {
      K.nameScope('foo', exceptionFn);
    };
    expect(nameScopeFn).toThrowError(exceptionValue);
  });
});

describe('getUID ', () => {
  it('second UID is different.', () => {
    const name = 'def';
    const firstUID = K.getUid(name);
    const secondUID = K.getUid(name);
    expect(secondUID).not.toEqual(firstUID);
  });

  it('with no prefix works and returns different UIDs.', () => {
    const firstUID = K.getUid();
    const secondUID = K.getUid();
    expect(firstUID).not.toEqual(secondUID);
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
    const result = K.categoricalCrossentropy(target, x, true);
    expectTensorsClose(result, expected);
  });

  it('from softmax', () => {
    const x = tensor2d([[0.3, 0.7], [0.4, 0.6]], [2, 2]);
    const target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
    const expected = tensor1d([
      -1 * (Math.log(0.3) * 0.25 + Math.log(0.7) * 0.75),
      -1 * (Math.log(0.4) * 0.1 + Math.log(0.6) * 0.9)
    ]);
    const result = K.categoricalCrossentropy(target, x, false);
    expectTensorsClose(result, expected);
  });
});

describeMathCPUAndGPU('sparseCategoricalCrossentropy ', () => {
  it('from logits', () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
    const target = tensor1d([0, 2]);
    const expected = tensor1d([
      -1 * Math.log(Math.exp(1) / (Math.exp(1) + Math.exp(2) + Math.exp(3))),
      -1 * Math.log(Math.exp(6) / (Math.exp(4) + Math.exp(5) + Math.exp(6)))
    ]);
    const result = K.sparseCategoricalCrossentropy(target, x, true);
    expectTensorsClose(result, expected);
  });

  it('from softmax', () => {
    const x = tensor2d([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]], [2, 3]);
    const target = tensor1d([0, 2]);
    const expected = tensor1d([-1 * Math.log(0.1), -1 * Math.log(0.5)]);
    const result = K.sparseCategoricalCrossentropy(target, x, false);
    expectTensorsClose(result, expected);
  });
});

describeMathCPUAndGPU('binaryCrossentropy', () => {
  function _binaryCrossentropy(target: Tensor, output: Tensor): Tensor {
    const targetComplement = K.scalarPlusArray(scalar(1), tfc.neg(target));
    const outputComplement = K.scalarPlusArray(scalar(1), tfc.neg(output));
    return tfc.neg(tfc.add(
      tfc.mul(target, tfc.log(output)),
      tfc.mul(targetComplement, tfc.log(outputComplement))));
  }

  it('from logits', () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
    const sigmoidX = tfc.sigmoid(x);
    const expected = _binaryCrossentropy(target, sigmoidX);
    const result = K.binaryCrossentropy(target, x, true);
    expectTensorsClose(result, expected);
  });

  it('from sigmoid', () => {
    const x = tensor2d([[0.3, 0.7], [0.4, 0.6]], [2, 2]);
    const target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
    const expected = _binaryCrossentropy(target, x);
    const result = K.binaryCrossentropy(target, x, false);
    expectTensorsClose(result, expected);
  });
});

describeMathCPUAndGPU('sigmoidCrossEntropyWithLogits', () => {
  it('outputs sigmoid cross-entropy', () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
    const targetComplement = K.scalarPlusArray(scalar(1), tfc.neg(target));
    const sigmoidX = tfc.sigmoid(x);
    const sigmoidXComplement = K.scalarPlusArray(scalar(1), tfc.neg(sigmoidX));
    const expected = tfc.add(
      tfc.mul(target, tfc.neg(tfc.log(sigmoidX))),
      tfc.mul(targetComplement, tfc.neg(tfc.log(sigmoidXComplement))));
    const result = K.sigmoidCrossEntropyWithLogits(target, x);
    expectTensorsClose(result, expected);
  });
});

describeMathCPUAndGPU('Sigmoid', () => {
  it('2D', () => {
    const xValues = [-5, -2, 0, 1, 2, 5];
    const x = tensor2d(xValues, [2, 3]);
    const y = tfc.sigmoid(x);
    const yValuesExpected = xValues.map(v => 1 / (1 + Math.exp(-v)));
    expectTensorsClose(y, tensor2d(yValuesExpected, [2, 3]));
  });
});

describeMathCPUAndGPU('hardSigmoid', () => {
  it('2D', () => {
    const xValues = [-5, -2, 0, 1, 2, 5];
    const x = tensor2d(xValues, [2, 3]);
    const y = K.hardSigmoid(x);
    const yValuesExpected = xValues.map(x => {
      const y = 0.2 * x + 0.5;
      if (y > 1) {
        return 1;
      } else if (y < 0) {
        return 0;
      } else {
        return y;
      }
    });
    expectTensorsClose(y, tensor2d(yValuesExpected, [2, 3]));
  });
});

describe('inTrainPhase', () => {
  it('training = true', () => {
    expect(K.inTrainPhase(() => -5, () => 5, true)).toEqual(-5);
  });
  it('training = false', () => {
    expect(K.inTrainPhase(() => -5, () => 5, false)).toEqual(5);
  });
  it('training = default false', () => {
    expect(K.inTrainPhase(() => -5, () => 5)).toEqual(5);
  });
});

describeMathCPUAndGPU('gradients', () => {
  it('Simple mean: 1 variable', () => {
    const var1 =
      new LayerVariable(K.scalarTimesArray(scalar(2.0), tfc.ones([2, 2])));
    const gradients = K.gradients(
      () => tfc.mean(var1.read()) as Scalar, [var1]);
    expect(gradients.length).toEqual(1);
    expectTensorsClose(
      tensor2d([[0.25, 0.25], [0.25, 0.25]], [2, 2]), gradients[0]);
  });
  it('Simple matmul and mean: 2 variables', () => {
    const var1 = new LayerVariable(tensor2d([[1, 0], [0, 0]], [2, 2]));
    const var2 = new LayerVariable(tensor2d([[1, 0], [0, 1]], [2, 2]));
    const gradients = K.gradients(
      () => tfc.mean(K.dot(var1.read(), var2.read())) as Scalar, [var1, var2]);
    expect(gradients.length).toEqual(2);
    // d(loss) / d(var1).
    expectTensorsClose(
      tensor2d([[0.25, 0.25], [0.25, 0.25]], [2, 2]), gradients[0]);
    // d(loss) / d(var2).
    expectTensorsClose(tensor2d([[0.25, 0.25], [0, 0]], [2, 2]), gradients[1]);
  });
});
