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
import {Scalar, scalar, Tensor, tensor1d, tensor2d, tensor3d, tensor4d, Tensor4D, zeros} from '@tensorflow/tfjs-core';
import * as _ from 'underscore';

import {DataFormat, PaddingMode, PoolMode} from '../common';
import {ConcreteTensor, DType, LayerVariable, SymbolicTensor} from '../types';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import * as K from './tfjs_backend';

// tslint:enable

const CT = ConcreteTensor;

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
    expect(K.shape(new CT(x))).toEqual([]);
  });

  it('Tensor1D', () => {
    const x = zeros([3]);
    expect(K.shape(x)).toEqual([3]);
    expect(K.shape(new CT(x))).toEqual([3]);
  });

  it('Tensor2D', () => {
    const x = zeros([3, 2]);
    expect(K.shape(x)).toEqual([3, 2]);
    expect(K.shape(new CT(x))).toEqual([3, 2]);
  });

  it('Tensor3D', () => {
    const x = zeros([4, 3, 2]);
    expect(K.shape(x)).toEqual([4, 3, 2]);
    expect(K.shape(new CT(x))).toEqual([4, 3, 2]);
  });

  it('Tensor4D', () => {
    const x = zeros([4, 3, 2, 1]);
    expect(K.shape(x)).toEqual([4, 3, 2, 1]);
    expect(K.shape(new CT(x))).toEqual([4, 3, 2, 1]);
  });
});

describe('intShape', () => {
  it('Scalar', () => {
    const x = zeros([]);
    expect(K.intShape(x)).toEqual([]);
    expect(K.intShape(new CT(x))).toEqual([]);
  });

  it('Tensor1D', () => {
    const x = zeros([3]);
    expect(K.intShape(x)).toEqual([3]);
    expect(K.intShape(new CT(x))).toEqual([3]);
  });

  it('Tensor2D', () => {
    const x = zeros([3, 2]);
    expect(K.intShape(x)).toEqual([3, 2]);
    expect(K.intShape(new CT(x))).toEqual([3, 2]);
  });

  it('Tensor3D', () => {
    const x = zeros([4, 3, 2]);
    expect(K.intShape(x)).toEqual([4, 3, 2]);
    expect(K.intShape(new CT(x))).toEqual([4, 3, 2]);
  });

  it('Tensor4D', () => {
    const x = zeros([4, 3, 2, 1]);
    expect(K.intShape(x)).toEqual([4, 3, 2, 1]);
    expect(K.intShape(new CT(x))).toEqual([4, 3, 2, 1]);
  });
});

describe('ndim', () => {
  it('Scalar', () => {
    const x = zeros([]);
    expect(K.ndim(x)).toEqual(0);
    expect(K.ndim(new CT(x))).toEqual(0);
  });

  it('Tensor1D', () => {
    const x = zeros([3]);
    expect(K.ndim(x)).toEqual(1);
    expect(K.ndim(new CT(x))).toEqual(1);
  });

  it('Tensor2D', () => {
    const x = zeros([3, 2]);
    expect(K.ndim(x)).toEqual(2);
    expect(K.ndim(new CT(x))).toEqual(2);
  });

  it('Tensor3D', () => {
    const x = zeros([4, 3, 2]);
    expect(K.ndim(x)).toEqual(3);
    expect(K.ndim(new CT(x))).toEqual(3);
  });

  it('Tensor4D', () => {
    const x = zeros([4, 3, 2, 1]);
    expect(K.ndim(x)).toEqual(4);
    expect(K.ndim(new CT(x))).toEqual(4);
  });
});

describe('dtype', () => {
  it('returns float32 for an Tensor', () => {
    const x = zeros([1]);
    expect(K.dtype(x)).toEqual(DType.float32);
  });

  it('returns float32 for a SymbolicTensor', () => {
    const x = new SymbolicTensor(DType.float32, [1], null, [], {});
    expect(K.dtype(x)).toEqual(DType.float32);
  });
});

describe('normalizeAxis', () => {
  let x: Tensor;
  beforeEach(() => {
    x = zeros([2, 2, 2, 2]);
  });

  it('handles single-value axis.', () => {
    expect(K.normalizeAxis(x, -1)).toEqual(3);
  });

  it('handles an array of axes.', () => {
    expect(K.normalizeAxis(x, [-1, 1])).toEqual([3, 1]);
  });

  it('returns null if axis is null.', () => {
    expect(K.normalizeAxis(x, null)).toBeNull();
  });

  it('throws exception if a single index is invalid.', () => {
    expect(() => K.normalizeAxis(x, 4)).toThrowError();
  });

  it('throws exception if a single index in an array is invalid.', () => {
    expect(() => K.normalizeAxis(x, [1, 4])).toThrowError();
  });

  it('throws exception if a single index in an array is null.', () => {
    expect(() => K.normalizeAxis(x, [null])).toThrowError();
  });
});

describeMathCPU('countParams', () => {
  it('Scalar', () => {
    const x = new CT(zeros([]));
    expect(K.countParams(x)).toEqual(1);
    expect(K.countParams(new LayerVariable(x))).toEqual(1);
  });

  it('Tensor1D', () => {
    const x = new CT(zeros([3]));
    expect(K.countParams(x)).toEqual(3);
    expect(K.countParams(new LayerVariable(x))).toEqual(3);
  });

  it('Tensor2D', () => {
    const x = new CT(zeros([3, 2]));
    expect(K.countParams(x)).toEqual(6);
    expect(K.countParams(new LayerVariable(x))).toEqual(6);
  });

  it('Tensor3D', () => {
    const x = new CT(zeros([4, 3, 2]));
    expect(K.countParams(x)).toEqual(24);
    expect(K.countParams(new LayerVariable(x))).toEqual(24);
  });

  it('Tensor4D', () => {
    const x = new CT(zeros([4, 3, 2, 1]));
    expect(K.countParams(x)).toEqual(24);
    expect(K.countParams(new LayerVariable(x))).toEqual(24);
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

describeMathCPUAndGPU('Transpose', () => {
  it('2D 2x2 implicit perm', () => {
    const x = tensor2d([[1, 3], [-3, 7]], [2, 2]);
    expectTensorsClose(K.transpose(x), tensor2d([[1, -3], [3, 7]], [2, 2]));
  });

  it('2D 3x2 explicit perm', () => {
    const x = tensor2d([[1, 3], [-3, 7], [4, 2]], [3, 2]);
    expectTensorsClose(
        K.transpose(x, [1, 0]), tensor2d([[1, -3, 4], [3, 7, 2]], [2, 3]));
  });

  it('3D 2x2x2 explicit perm', () => {
    const x = tensor3d([[[1, 3], [-3, 7]], [[10, 11], [12, 13]]], [2, 2, 2]);
    expectTensorsClose(
        K.transpose(x, [1, 2, 0]),
        tensor3d([[[1, 10], [3, 11]], [[-3, 12], [7, 13]]], [2, 2, 2]));
  });
});

describeMathCPUAndGPU('Reverse', () => {
  it('1D, along first axis', () => {
    const x = tensor1d([1, 3, -3, 7]);
    expectTensorsClose(K.reverse(x, 0), tensor1d([7, -3, 3, 1]));
  });
  it('2D, along first axis', () => {
    const x = tensor2d([[1, 3], [-3, 7]], [2, 2]);
    expectTensorsClose(K.reverse(x, 0), tensor2d([[-3, 7], [1, 3]], [2, 2]));
  });
  it('2D, along second axis', () => {
    const x = tensor2d([[1, 3], [-3, 7]], [2, 2]);
    expectTensorsClose(K.reverse(x, 1), tensor2d([[3, 1], [7, -3]], [2, 2]));
    expectTensorsClose(K.reverse(x, -1), tensor2d([[3, 1], [7, -3]], [2, 2]));
  });
  it('2D, along both axes', () => {
    const x = tensor2d([[1, 3], [-3, 7]], [2, 2]);
    expectTensorsClose(
        K.reverse(x, [0, 1]), tensor2d([[7, -3], [3, 1]], [2, 2]));
    expectTensorsClose(
        K.reverse(x, [0, -1]), tensor2d([[7, -3], [3, 1]], [2, 2]));
  });
  it('3D', () => {
    const x = tensor3d([3, 7], [1, 2, 1]);
    expectTensorsClose(K.reverse(x, 0), tensor3d([3, 7], [1, 2, 1]));
    expectTensorsClose(K.reverse(x, 1), tensor3d([7, 3], [1, 2, 1]));
    expectTensorsClose(K.reverse(x, 2), tensor3d([3, 7], [1, 2, 1]));
    expectTensorsClose(K.reverse(x, -1), tensor3d([3, 7], [1, 2, 1]));
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

describeMathCPUAndGPU('squeeze', () => {
  it('1D to Scalar', () => {
    const value = 10;
    const xTensor = tensor1d([value]);
    expectTensorsClose(K.squeeze(xTensor, 0), scalar(value));
  });
  it('2D to 1D: Last dimension', () => {
    const x = tensor2d([[10], [20], [30]]);
    const y = tensor1d([10, 20, 30]);
    expectTensorsClose(K.squeeze(x, 1), y);
  });
  it('2D to 1D: First dimension', () => {
    const x = tensor2d([[10, 20, 30]]);
    const y = tensor1d([10, 20, 30]);
    expectTensorsClose(K.squeeze(x, 0), y);
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
  it('1D ConcreteTensor', () => {
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

  it('3D ConcreteTensor', () => {
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

  it('1D ConcreteTensor leads to error', () => {
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

  it('3D ConcreteTensor', () => {
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
    expectTensorsClose(y, K.ones([2, 3, 4]));
  });
  it('Mismatch in x dimensions and n length leads to exception', () => {
    expect(() => K.tile(K.zeros([2, 2]), 1))
        .toThrowError(/The length of input n \(1\) does not match .*2/);
  });
});

describeMathCPUAndGPU('Create Variable', () => {
  it('From Tensor, no explicit name', () => {
    const v = K.variable(zeros([2, 2]));
    expect(v.name.indexOf('Variable')).toEqual(0);
    expect(v.shape).toEqual([2, 2]);
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('From ConcreteTensor, no explicit name', () => {
    const v = K.variable(zeros([3]));
    expect(v.name.indexOf('Variable')).toEqual(0);
    expect(v.shape).toEqual([3]);
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('From ConcreteTensor, explicit name', () => {
    const v = K.variable(zeros([3]), undefined, 'Var1');
    expect(v.name.indexOf('Var1')).toEqual(0);
    expect(v.shape).toEqual([3]);
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
  });
});

describeMathCPUAndGPU('batchGetValue', () => {
  it('Legnth-3 Array, Mixed ConcreteTensor and Variable', () => {
    const v1 = K.variable(zeros([]));
    const v2 = K.variable(zeros([2]));
    const v3 = K.variable(zeros([2, 2]));
    const values = K.batchGetValue([v1, v2, v3]);
    expect(values.length).toEqual(3);
    expect(values[0].shape).toEqual([]);
    expect(values[0].dataSync()).toEqual(new Float32Array([0]));
    expect(values[1].shape).toEqual([2]);
    expect(values[1].dataSync()).toEqual(new Float32Array([0, 0]));
    expect(values[2].shape).toEqual([2, 2]);
    expect(values[2].dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });
});

describeMathCPUAndGPU('batchSetValue', () => {
  it('Update using Tensor values', () => {
    const v1 = K.randomUniformVariable([2], 0, 1);
    const v2 = K.randomUniformVariable([2, 2], 0, 1);
    K.batchSetValue([[v1, zeros([2])], [v2, zeros([2, 2])]]);
    expect(v1.shape).toEqual([2]);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0, 0]));
    expect(v2.shape).toEqual([2, 2]);
    expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('Update using ConcreteTensor values', () => {
    const v1 = K.randomUniformVariable([], 0, 1);
    const v2 = K.randomUniformVariable([2, 2, 1], 0, 1);
    K.batchSetValue([[v1, zeros([])], [v2, zeros([2, 2, 1])]]);
    expect(v1.shape).toEqual([]);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0]));
    expect(v2.shape).toEqual([2, 2, 1]);
    expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('Update empty Array', () => {
    K.batchSetValue([]);
  });
});

describeMathCPUAndGPU('ZerosVariable', () => {
  it('Scalar', () => {
    const s = K.zerosVariable([], DType.float32, 'Scalar');
    expect(s.name.indexOf('Scalar')).toEqual(0);
    expect(K.shape(s)).toEqual([]);
    expect(s.read().dataSync()).toEqual(new Float32Array([0]));
  });

  it('Vector', () => {
    const v = K.zerosVariable([3], DType.float32, 'Vector');
    expect(v.name.indexOf('Vector')).toEqual(0);
    expect(K.shape(v)).toEqual([3]);
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('Matrix', () => {
    const m = K.zerosVariable([2, 2], DType.float32, 'Matrix');
    expect(m.name.indexOf('Matrix')).toEqual(0);
    expect(K.shape(m)).toEqual([2, 2]);
    expect(m.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('3D', () => {
    const t = K.zerosVariable([2, 2, 2], DType.float32, 'Tertiary');
    expect(t.name.indexOf('Tertiary')).toEqual(0);
    expect(K.shape(t)).toEqual([2, 2, 2]);
    expect(t.read().dataSync()).toEqual(new Float32Array([
      0, 0, 0, 0, 0, 0, 0, 0
    ]));
  });

  it('4D', () => {
    const q = K.zerosVariable([1, 2, 1, 3], DType.float32, 'Quaternary');
    expect(q.name.indexOf('Quaternary')).toEqual(0);
    expect(K.shape(q)).toEqual([1, 2, 1, 3]);
    expect(q.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
  });
});

describeMathCPUAndGPU('OnesVariable', () => {
  it('Scalar', () => {
    const s = K.onesVariable([], DType.float32, 'Scalar');
    expect(s.name.indexOf('Scalar')).toEqual(0);
    expect(K.shape(s)).toEqual([]);
    expect(s.read().dataSync()).toEqual(new Float32Array([1]));
  });
  it('Vector', () => {
    const v = K.onesVariable([3], DType.float32, 'Vector');
    expect(v.name.indexOf('Vector')).toEqual(0);
    expect(K.shape(v)).toEqual([3]);
    expect(v.read().dataSync()).toEqual(new Float32Array([1, 1, 1]));
  });
  it('Matrix', () => {
    const m = K.onesVariable([2, 2], DType.float32, 'Matrix');
    expect(m.name.indexOf('Matrix')).toEqual(0);
    expect(K.shape(m)).toEqual([2, 2]);
    expect(m.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1]));
  });
  it('3D', () => {
    const t = K.onesVariable([2, 2, 2], DType.float32, 'Tertiary');
    expect(t.name.indexOf('Tertiary')).toEqual(0);
    expect(K.shape(t)).toEqual([2, 2, 2]);
    expect(t.read().dataSync()).toEqual(new Float32Array([
      1, 1, 1, 1, 1, 1, 1, 1
    ]));
  });
  it('4D', () => {
    const q = K.onesVariable([1, 2, 1, 3], DType.float32, 'Quaternary');
    expect(q.name.indexOf('Quaternary')).toEqual(0);
    expect(K.shape(q)).toEqual([1, 2, 1, 3]);
    expect(q.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1, 1, 1]));
  });
});

describeMathCPUAndGPU('ZerosLike', () => {
  it('Scalar', () => {
    const s = K.zerosLike(K.randomUniform([], -10, 10));
    expect(K.shape(s)).toEqual([]);
    expect(s.read().dataSync()).toEqual(new Float32Array([0]));
  });

  it('Vector', () => {
    const v = K.zerosLike(K.randomUniform([3], -10, 10));
    expect(K.shape(v)).toEqual([3]);
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('Matrix', () => {
    const m = K.zerosLike(K.randomUniform([2, 2], -10, 10));
    expect(K.shape(m)).toEqual([2, 2]);
    expect(m.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('3D', () => {
    const t = K.zerosLike(K.randomUniform([2, 2, 2], -10, 10));
    expect(K.shape(t)).toEqual([2, 2, 2]);
    expect(t.read().dataSync()).toEqual(new Float32Array([
      0, 0, 0, 0, 0, 0, 0, 0
    ]));
  });

  it('4D', () => {
    const q = K.zerosLike(K.randomUniform([1, 2, 1, 3], -10, 10));
    expect(K.shape(q)).toEqual([1, 2, 1, 3]);
    expect(q.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
  });
});

describeMathCPUAndGPU('OnesLike', () => {
  it('Scalar', () => {
    const s = K.onesLike(K.randomUniform([], -10, 10));
    expect(K.shape(s)).toEqual([]);
    expect(s.read().dataSync()).toEqual(new Float32Array([1]));
  });

  it('Vector', () => {
    const v = K.onesLike(K.randomUniform([3], -10, 10));
    expect(K.shape(v)).toEqual([3]);
    expect(v.read().dataSync()).toEqual(new Float32Array([1, 1, 1]));
  });

  it('Matrix', () => {
    const m = K.onesLike(K.randomUniform([2, 2], -10, 10));
    expect(K.shape(m)).toEqual([2, 2]);
    expect(m.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1]));
  });

  it('3D', () => {
    const t = K.onesLike(K.randomUniform([2, 2, 2], -10, 10));
    expect(K.shape(t)).toEqual([2, 2, 2]);
    expect(t.read().dataSync()).toEqual(new Float32Array([
      1, 1, 1, 1, 1, 1, 1, 1
    ]));
  });

  it('4D', () => {
    const q = K.onesLike(K.randomUniform([1, 2, 1, 3], -10, 10));
    expect(K.shape(q)).toEqual([1, 2, 1, 3]);
    expect(q.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1, 1, 1]));
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

describeMathCPUAndGPU('eye (I-matrix builder)', () => {
  it('Zero sized 2D matrix', () => {
    const I = K.eye(0);
    expect(I.shape).toEqual([0, 0]);
  });
  it('1 sized 2D matrix', () => {
    const I = K.eye(1);
    expect(I.shape).toEqual([1, 1]);
    expect(I.dataSync()).toEqual(new Float32Array([1]));
  });
  it('2 sized 2D matrix', () => {
    const I = K.eye(2);
    expect(I.shape).toEqual([2, 2]);
    expect(I.dataSync()).toEqual(new Float32Array([1, 0, 0, 1]));
  });
  it('Variable Zero sized 2D matrix', () => {
    const I = K.eyeVariable(0);
    expect(I.shape).toEqual([0, 0]);
  });
  it('Variable 1 sized 2D matrix', () => {
    const I = K.eyeVariable(1);
    expect(I.shape).toEqual([1, 1]);
    expect(I.read().dataSync()).toEqual(new Float32Array([1]));
  });
  it('Variable 2 sized 2D matrix', () => {
    const I = K.eyeVariable(2);
    expect(I.shape).toEqual([2, 2]);
    expect(I.read().dataSync()).toEqual(new Float32Array([1, 0, 0, 1]));
  });
});

describeMathCPUAndGPU('neg', () => {
  it('negates its tensor', () => {
    const x = tensor1d([-1, 0, 1]);
    const expected = tensor1d([1, 0, -1]);
    expectTensorsClose(K.neg(x), expected);
  });
});

describeMathCPUAndGPU('Add', () => {
  it('3D', () => {
    const shape = [2, 3, 4];
    const x = K.ones(shape);
    const y = K.ones(shape);
    expectTensorsClose(
        K.add(x, y), K.scalarTimesArray(scalar(2), K.ones(shape)));
  });
});

describeMathCPUAndGPU('subtract', () => {
  it('3D', () => {
    const shape = [2, 3, 4];
    const x = K.ones(shape);
    const y = K.ones(shape);
    expectTensorsClose(K.subtract(x, y), K.zeros(shape));
  });
});

describeMathCPUAndGPU('Multiply', () => {
  it('3D', () => {
    const shape = [2, 3, 4];
    const x = K.scalarTimesArray(scalar(4), K.ones(shape));
    const y = K.scalarTimesArray(scalar(5), K.ones(shape));
    expectTensorsClose(
        K.multiply(x, y), K.scalarTimesArray(scalar(20), K.ones(shape)));
  });
});

describeMathCPUAndGPU('divide', () => {
  it('3D', () => {
    const shape = [2, 3, 4];
    const x = K.scalarTimesArray(scalar(4), K.ones(shape));
    const y = K.scalarTimesArray(scalar(4), K.ones(shape));
    expectTensorsClose(K.divide(x, y), K.ones(shape));
  });
});

describeMathCPUAndGPU('scalarTimesArray', () => {
  it('Scalar x Scalar', () => {
    expectTensorsClose(K.scalarTimesArray(scalar(-2), scalar(-3)), scalar(6));
  });
  it('Scalar x 4D', () => {
    const y = K.scalarTimesArray(scalar(-2), K.ones([2, 2, 2, 2]));
    expect(y.shape).toEqual([2, 2, 2, 2]);
    const yValues = Array.from(y.dataSync());
    expect(_.uniq(yValues)).toEqual([-2]);
  });
});

describeMathCPUAndGPU('scalarPlusArray', () => {
  it('Scalar + Scalar', () => {
    expectTensorsClose(K.scalarPlusArray(scalar(-2), scalar(-3)), scalar(-5));
  });
  it('Scalar + 4D', () => {
    const shape = [2, 2, 2, 2];
    const y = K.scalarPlusArray(scalar(-1), K.ones(shape));
    expectTensorsClose(y, K.zeros(shape));
  });
});

describeMathCPUAndGPU('randomUniform', () => {
  it('Scalar', () => {
    const s = K.randomUniform([], -10, 10);
    expect(K.shape(s)).toEqual([]);
    expect(s.dataSync()[0]).toBeGreaterThanOrEqual(-10);
    expect(s.dataSync()[0]).toBeLessThanOrEqual(10);
  });

  it('1D', () => {
    const v = K.randomUniform([20], -10, 10);
    expect(K.shape(v)).toEqual([20]);
    const vValuesSorted = v.dataSync().sort();
    expect(vValuesSorted[0]).toBeGreaterThanOrEqual(-10);
    expect(vValuesSorted[vValuesSorted.length - 1]).toBeLessThanOrEqual(10);
  });

  it('2D', () => {
    const x = K.randomUniform([3, 20], 100, 200);
    expect(K.shape(x)).toEqual([3, 20]);
    const xValuesSorted = x.dataSync().sort();
    expect(xValuesSorted[0]).toBeGreaterThanOrEqual(100);
    expect(xValuesSorted[xValuesSorted.length - 1]).toBeLessThanOrEqual(200);
  });

  it('3D', () => {
    const y = K.randomUniform([2, 3, 4], -100, -50);
    expect(K.shape(y)).toEqual([2, 3, 4]);
    const yValuesSorted = y.dataSync().sort();
    expect(yValuesSorted[0]).toBeGreaterThanOrEqual(-100);
    expect(yValuesSorted[yValuesSorted.length - 1]).toBeLessThanOrEqual(-50);
  });
});

describeMathCPUAndGPU('truncatedNormal', () => {
  it('Scalar', () => {
    const s = K.truncatedNormal([], 0, 10);
    expect(K.shape(s)).toEqual([]);
    expect(s.dataSync()[0]).toBeGreaterThan(-20);
    expect(s.dataSync()[0]).toBeLessThan(20);
  });

  it('1D', () => {
    const v = K.truncatedNormal([20], 0, 2);
    expect(K.shape(v)).toEqual([20]);
    const vValuesSorted = v.dataSync().sort();
    expect(vValuesSorted[0]).toBeGreaterThan(-4);
    expect(vValuesSorted[vValuesSorted.length - 1]).toBeLessThan(4);
  });

  it('2D', () => {
    const x = K.truncatedNormal([3, 20], -10, 20);
    expect(K.shape(x)).toEqual([3, 20]);
    const xValuesSorted = x.dataSync().sort();
    expect(xValuesSorted[0]).toBeGreaterThan(-50);
    expect(xValuesSorted[xValuesSorted.length - 1]).toBeLessThan(30);
  });

  it('3D', () => {
    const y = K.truncatedNormal([2, 3, 4], 100, 10);
    expect(K.shape(y)).toEqual([2, 3, 4]);
    const yValuesSorted = y.dataSync().sort();
    expect(yValuesSorted[0]).toBeGreaterThan(80);
    expect(yValuesSorted[yValuesSorted.length - 1]).toBeLessThan(120);
  });
});

describeMathCPUAndGPU('randomNormal', () => {
  const dtypes = [DType.float32, DType.int32];
  for (const dtype of dtypes) {
    // TODO(bileschi): Add probabilistic assertions on values here.
    it(`Scalar ${dtype}`, () => {
      const s = K.randomNormal([], 0, 10, dtype);
      expect(K.shape(s)).toEqual([]);
    });

    it(`1D ${dtype}`, () => {
      const v = K.randomNormal([20], 0, 2, dtype);
      expect(K.shape(v)).toEqual([20]);
    });

    it(`2D ${dtype}`, () => {
      const x = K.randomNormal([3, 20], -10, 20, dtype);
      expect(K.shape(x)).toEqual([3, 20]);
    });

    it(`3D ${dtype}`, () => {
      const y = K.randomNormal([2, 3, 4], 100, 10, dtype);
      expect(K.shape(y)).toEqual([2, 3, 4]);
    });
  }
});

describeMathCPUAndGPU('Variable update', () => {
  it('Update', () => {
    const v = new LayerVariable(scalar(10.0));
    K.update(v, scalar(20.0));
    expectTensorsClose(v.read(), scalar(20.0));
  });
  it('Update: Incompatible shape', () => {
    const v = new LayerVariable(tensor1d([10.0, 20.0]));
    const x = tensor1d([10.0, 20.0, 30.0]);
    expect(() => K.update(v, x)).toThrowError();
  });
  it('UpdateAdd', () => {
    const v = new LayerVariable(scalar(10.0));
    K.updateAdd(v, scalar(20.0));
    expectTensorsClose(v.read(), scalar(30.0));
  });
  it('UpdateAdd: Incompatible shape', () => {
    const v = new LayerVariable(tensor1d([10.0, 20.0]));
    const x = tensor1d([0.0, 10.0, 20.0]);
    expect(() => K.updateAdd(v, x)).toThrowError();
  });
  it('UpdateSub', () => {
    const v = new LayerVariable(scalar(10.0));
    K.updateSub(v, scalar(20.0));
    const vNew = v.read();
    expectTensorsClose(vNew, scalar(-10.0));
  });
  it('UpdateSub: Incompatible shape', () => {
    const v = new LayerVariable(tensor1d([10.0, 20.0]));
    const x = tensor1d([0.0, 10.0, 20.0]);
    expect(() => K.updateSub(v, x)).toThrowError();
  });
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

describeMathCPUAndGPU('Mean', () => {
  it('reduce_mean', () => {
    expectTensorsClose(
        K.mean(tensor2d([[4, -1], [0, -2]], [2, 2])), scalar(0.25));
  });
  it('mean 2D, axis=1, keepdims=false', () => {
    expectTensorsClose(
        K.mean(tensor2d([[4, -1], [0, -2]], [2, 2]), 1), tensor1d([1.5, -1]));
  });
  it('mean 2D, axis=1, keepdims=true', () => {
    expectTensorsClose(
        K.mean(tensor2d([[4, -1], [0, -2]], [2, 2]), 1, true),
        tensor2d([[1.5], [-1]], [2, 1]));
  });
  it('mean 3D, axis=[1,2], keepdims=false', () => {
    expectTensorsClose(
        K.mean(
            tensor3d([[[4, -1], [0, -2]], [[40, -10], [0, -20]]], [2, 2, 2]),
            [1, 2]),
        tensor1d([0.25, 2.5]));
  });
  it('mean 3D, axis=[1,2], keepdims=true', () => {
    expectTensorsClose(
        K.mean(
            tensor3d([[[4, -1], [0, -2]], [[40, -10], [0, -20]]], [2, 2, 2]),
            [1, 2], true),
        tensor3d([[[0.25]], [[2.5]]], [2, 1, 1]));
  });
  it('reduce_mean keepdims=true', () => {
    expectTensorsClose(
        K.mean(tensor2d([[4, -1], [0, -2]], [2, 2]), undefined, true),
        tensor2d([[0.25]], [1, 1]));
  });
});

describeMathCPUAndGPU('Argmax', () => {
  it('2D, default axis', () => {
    expectTensorsClose(
        K.argmax(tensor2d([[4, -1], [-2, 0]], [2, 2])),
        tensor1d([0, 1], 'int32'));
  });
  it('2D, axis=-1', () => {
    expectTensorsClose(
        K.argmax(tensor2d([[4, -1], [-2, 0]], [2, 2]), -1),
        tensor1d([0, 1], 'int32'));
  });
  it('2D, axis=0', () => {
    expectTensorsClose(
        K.argmax(tensor2d([[4, -1], [3, 2]], [2, 2]), 0),
        tensor1d([0, 1], 'int32'));
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

describeMathCPUAndGPU('Max', () => {
  it('reduce_max', () => {
    expectTensorsClose(K.max(tensor2d([[4, -1], [0, -2]], [2, 2])), scalar(4));
  });
  it('max 2D, axis=1, keepdims=false', () => {
    expectTensorsClose(
        K.max(tensor2d([[4, -1], [0, -2]], [2, 2]), 1), tensor1d([4, 0]));
  });
  it('max 2D, axis=1, keepdims=true', () => {
    expectTensorsClose(
        K.max(tensor2d([[4, -1], [0, -2]], [2, 2]), 1, true),
        tensor2d([[4], [0]], [2, 1]));
  });
  it('max 3D, axis=[1,2], keepdims=false', () => {
    expectTensorsClose(
        K.max(
            tensor3d([[[4, -1], [0, -2]], [[40, -10], [0, -20]]], [2, 2, 2]),
            [1, 2]),
        tensor1d([4, 40]));
  });
  it('max 3D, axis=[1,2], keepdims=true', () => {
    expectTensorsClose(
        K.max(
            tensor3d([[[4, -1], [0, -2]], [[40, -10], [0, -20]]], [2, 2, 2]),
            [1, 2], true),
        tensor3d([[[4]], [[40]]], [2, 1, 1]));
  });
  it('reduce_max keepdims=true', () => {
    expectTensorsClose(
        K.max(tensor2d([[4, -1], [0, -2]], [2, 2]), undefined, true),
        tensor2d([[4]], [1, 1]));
  });
});

describeMathCPUAndGPU('Min', () => {
  it('reduce_min', () => {
    expectTensorsClose(K.min(tensor2d([[4, -1], [0, -2]], [2, 2])), scalar(-2));
  });
  it('min 2D, axis=1, keepdims=false', () => {
    expectTensorsClose(
        K.min(tensor2d([[4, -1], [0, -2]], [2, 2]), 1), tensor1d([-1, -2]));
  });
  it('min 2D, axis=1, keepdims=true', () => {
    expectTensorsClose(
        K.min(tensor2d([[4, -1], [0, -2]], [2, 2]), 1, true),
        tensor2d([[-1], [-2]], [2, 1]));
  });
  it('min 3D, axis=[1,2], keepdims=false', () => {
    expectTensorsClose(
        K.min(
            tensor3d([[[4, -1], [0, -2]], [[40, -10], [0, -20]]], [2, 2, 2]),
            [1, 2]),
        tensor1d([-2, -20]));
  });
  it('min 3D, axis=[1,2], keepdims=true', () => {
    expectTensorsClose(
        K.min(
            tensor3d([[[4, -1], [0, -2]], [[40, -10], [0, -20]]], [2, 2, 2]),
            [1, 2], true),
        tensor3d([[[-2]], [[-20]]], [2, 1, 1]));
  });
  it('reduce_min keepdims=true', () => {
    expectTensorsClose(
        K.min(tensor2d([[4, -1], [0, -2]], [2, 2]), undefined, true),
        tensor2d([[-2]], [1, 1]));
  });
});

describeMathCPUAndGPU('Sum', () => {
  it('reduce_sum', () => {
    expectTensorsClose(K.sum(tensor2d([[4, -1], [0, -2]], [2, 2])), scalar(1));
  });
  it('sum 2D, axis=1, keepdims=false', () => {
    expectTensorsClose(
        K.sum(tensor2d([[4, -1], [0, -2]], [2, 2]), 1), tensor1d([3, -2]));
  });
  it('sum 2D, axis=1, keepdims=true', () => {
    expectTensorsClose(
        K.sum(tensor2d([[4, -1], [0, -2]], [2, 2]), 1, true),
        tensor2d([[3], [-2]], [2, 1]));
  });
  it('sum 3D, axis=[1,2], keepdims=false', () => {
    expectTensorsClose(
        K.sum(
            tensor3d([[[4, -1], [0, -2]], [[40, -10], [0, -20]]], [2, 2, 2]),
            [1, 2]),
        tensor1d([1, 10]));
  });
  it('sum 3D, axis=[1,2], keepdims=true', () => {
    expectTensorsClose(
        K.sum(
            tensor3d([[[4, -1], [0, -2]], [[40, -10], [0, -20]]], [2, 2, 2]),
            [1, 2], true),
        tensor3d([[[1]], [[10]]], [2, 1, 1]));
  });
  it('reduce_sum keepdims=true', () => {
    expectTensorsClose(
        K.sum(tensor2d([[4, -1], [0, -2]], [2, 2]), undefined, true),
        tensor2d([[1]], [1, 1]));
  });
});

describeMathCPUAndGPU('Abs', () => {
  it('Element-wise abs', () => {
    expectTensorsClose(
        K.abs(tensor2d([[1, -1], [0, -2]], [2, 2])),
        tensor2d([1, 1, 0, 2], [2, 2]));
  });
});

describeMathCPUAndGPU('Square', () => {
  it('Element-wise square', () => {
    expectTensorsClose(
        K.square(tensor2d([[1, -2], [-3, 4]], [2, 2])),
        tensor2d([1, 4, 9, 16], [2, 2]));
  });
});

describeMathCPUAndGPU('Sqrt', () => {
  it('Element-wise sqrt', () => {
    expectTensorsClose(
        K.sqrt(tensor2d([[1, 4], [9, 16]], [2, 2])),
        tensor2d([1, 2, 3, 4], [2, 2]));
  });
});


describeMathCPUAndGPU('Exp', () => {
  it('Element-wise Exp', () => {
    expectTensorsClose(
        K.exp(tensor2d([[0, 1], [3, 2]], [2, 2])),
        tensor2d([Math.exp(0), Math.exp(1), Math.exp(3), Math.exp(2)], [2, 2]));
  });
});

describeMathCPUAndGPU('Log', () => {
  it('Element-wise log', () => {
    expectTensorsClose(
        K.log(tensor2d([[1, 1], [1, 1]], [2, 2])),
        tensor2d([0, 0, 0, 0], [2, 2]));
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

describeMathCPUAndGPU('Clip', () => {
  it('Element-wise Clip', () => {
    expectTensorsClose(
        K.clip(tensor2d([[-5, -2], [2, 5]], [2, 2]), -3, 3),
        tensor2d([-3, -2, 2, 3], [2, 2]));
  });
});

describeMathCPUAndGPU('Equal', () => {
  it('Element-wise 1D float32', () => {
    const x = tensor1d([1, 3, 3, 7]);
    const y = tensor1d([-1, -3, -3, 7]);
    const output = K.equal(x, y);
    expect(output.shape).toEqual([4]);
    expect(Array.from(output.dataSync())).toEqual([0, 0, 0, 1]);
  });
  it('Element-wise 2D int32', () => {
    const x = tensor2d([[0, 10], [20, 30]], [2, 2], 'int32');
    const y = tensor2d([[0, 11], [22, 30]], [2, 2], 'int32');
    const output = K.equal(x, y);
    expect(output.shape).toEqual([2, 2]);
    expect(Array.from(output.dataSync())).toEqual([1, 0, 0, 1]);
  });
});

describeMathCPUAndGPU('Greater', () => {
  it('Element-wise 1D float32', () => {
    const x = tensor1d([1, 3, 3, 7]);
    const y = tensor1d([0.9, 3, -3, 7.1]);
    const output = K.greater(x, y);
    expect(output.shape).toEqual([4]);
    expect(Array.from(output.dataSync())).toEqual([1, 0, 1, 0]);
  });
  it('Element-wise 2D int32', () => {
    const x = tensor2d([[0, 10], [20, 30]], [2, 2], 'int32');
    const y = tensor2d([[0, 11], [19, 30]], [2, 2], 'int32');
    const output = K.greater(x, y);
    expect(output.shape).toEqual([2, 2]);
    expect(Array.from(output.dataSync())).toEqual([0, 0, 1, 0]);
  });
});

describeMathCPUAndGPU('GreaterEqual', () => {
  it('Element-wise 1D float32', () => {
    const x = tensor1d([1, 3, 3, 7]);
    const y = tensor1d([0.9, 3, -3, 7.1]);
    const output = K.greaterEqual(x, y);
    expect(output.shape).toEqual([4]);
    expect(Array.from(output.dataSync())).toEqual([1, 1, 1, 0]);
  });
  it('Element-wise 2D int32', () => {
    const x = tensor2d([[0, 10], [20, 30]], [2, 2], 'int32');
    const y = tensor2d([[0, 11], [19, 30]], [2, 2], 'int32');
    const output = K.greaterEqual(x, y);
    expect(output.shape).toEqual([2, 2]);
    expect(Array.from(output.dataSync())).toEqual([1, 0, 1, 1]);
  });
});

describeMathCPUAndGPU('maximum', () => {
  it('Element-wise maximum', () => {
    expectTensorsClose(
        K.maximum(
            tensor2d([[0, 1], [1, -1]], [2, 2]),
            tensor2d([[1, 0], [1, 1]], [2, 2])),
        K.ones([2, 2]));
  });
});

describeMathCPUAndGPU('minimum', () => {
  it('Element-wise minimum', () => {
    expectTensorsClose(
        K.minimum(
            tensor2d([[0, 1], [1, -1]], [2, 2]),
            tensor2d([[1, 0], [1, 1]], [2, 2])),
        new CT(tensor2d([[0, 0], [1, -1]], [2, 2])).value());
  });
  it('Broadcast element-wise minimum', () => {
    expectTensorsClose(
        K.minimum(tensor2d([[0, 1], [1, -1]], [2, 2]), tensor1d([0.0])),
        new CT(tensor2d([[0, 0], [0, -1]], [2, 2])).value());
  });
});

describeMathCPUAndGPU('Sin', () => {
  it('Element-wise sin', () => {
    expectTensorsClose(
        K.sin(new CT(tensor2d([[1, 2], [3, 4]], [2, 2]))),
        tensor2d([Math.sin(1), Math.sin(2), Math.sin(3), Math.sin(4)], [2, 2]));
  });
});

describeMathCPUAndGPU('Cos', () => {
  it('Element-wise cos', () => {
    expectTensorsClose(
        K.cos(new CT(tensor2d([[1, 2], [3, 4]], [2, 2]))),
        tensor2d([Math.cos(1), Math.cos(2), Math.cos(3), Math.cos(4)], [2, 2]));
  });
});

describeMathCPUAndGPU('softsign', () => {
  it('Element-wise softsign', () => {
    expectTensorsClose(
        K.tanh(tensor2d([[-2, -1], [1, 2]], [2, 2])),
        tensor2d(
            [Math.tanh(-2), Math.tanh(-1), Math.tanh(1), Math.tanh(2)],
            [2, 2]));
  });
});

describeMathCPUAndGPU('Hyperbolic tan', () => {
  it('Element-wise tanh', () => {
    expectTensorsClose(
        K.tanh(tensor2d([[-2, -1], [1, 2]], [2, 2])),
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
      const x = tensor2d(_.range(1, 21), [10, 2]);
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
    const x = K.ones([2]);
    const y = tensor1d([-1, 1]);
    expectTensorsClose(K.biasAdd(x, y), tensor1d([0, 2]));
  });
  it('2D + 1D', () => {
    const x = K.ones([2, 2]);
    const y = tensor1d([-1, 1]);
    expectTensorsClose(K.biasAdd(x, y), tensor2d([[0, 2], [0, 2]], [2, 2]));
  });
  it('3D + 1D', () => {
    const x = K.ones([2, 2, 2]);
    const y = tensor1d([-1, 1]);
    expectTensorsClose(
        K.biasAdd(x, y),
        tensor3d([[[0, 2], [0, 2]], [[0, 2], [0, 2]]], [2, 2, 2]));
  });
  it('4D + 1D', () => {
    const x = K.ones([1, 2, 2, 2]);
    const y = tensor1d([-1, 1]);
    expectTensorsClose(
        K.biasAdd(x, y),
        tensor4d([[[[0, 2], [0, 2]], [[0, 2], [0, 2]]]], [1, 2, 2, 2]));
  });
  it('2D + 1D: Incompatible size', () => {
    const x = K.ones([2, 2]);
    const y = tensor1d([-1, 0, 1]);
    expect(() => K.biasAdd(x, y)).toThrowError();
  });
  it('3D + 2D leads to error', () => {
    const x = K.ones([2, 2, 2]);
    const y = K.ones([2, 2]);
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

describeMathCPUAndGPU('selu', () => {
  it('selu', () => {
    const alpha = 1.6732632423543772848170429916717;
    const scale = 1.0507009873554804934193349852946;
    const xData = [-1, 0, 1, -1];
    expectTensorsClose(
        K.selu(tensor2d(xData, [2, 2])),
        tensor2d(
            xData.map(x => scale * (x < 0 ? (alpha * (Math.exp(x) - 1)) : x)),
            [2, 2]));
  });
});


describeMathCPUAndGPU('ReLU', () => {
  it('ReLU', () => {
    expectTensorsClose(
        K.relu(tensor2d([[-1, 1], [1, -1]], [2, 2])),
        tensor2d([0, 1, 1, 0], [2, 2]));
    expectTensorsClose(
        K.relu(tensor2d([[1, -1], [-1, 1]], [2, 2])),
        tensor2d([1, 0, 0, 1], [2, 2]));
  });
});

describeMathCPUAndGPU('softplus', () => {
  it('softplus', () => {
    const xData = [-1, 0, 1, -1];
    expectTensorsClose(
        K.softplus(tensor2d(xData, [2, 2])),
        tensor2d(xData.map(x => Math.log(Math.exp(x) + 1)), [2, 2]));
  });
});

describeMathCPUAndGPU('softsign', () => {
  it('softsign', () => {
    const xData = [-1, 0, 1, -1];
    expectTensorsClose(
        K.softsign(tensor2d(xData, [2, 2])),
        tensor2d(xData.map(x => x / (Math.abs(x) + 1)), [2, 2]));
  });
});

describeMathCPUAndGPU('conv1dWithBias', () => {
  const xLength4Data = [10, 20, 40, 80];
  const kernelLength2Data = [1, -1];
  const biasScalarData = 2.2;
  // In the basic case, this convolves [10, 20, 40, 80] with the kernel [1, -1],
  // producing [-10, -20, -40], and adds the bias 2.2, producing
  // [-7.8, -17.8, -37.7].  The test is reproduced for either 1 or 2 output
  // channels, and several reasonable data formats.

  const outChannelsArray = [1, 2];
  const dataFormats: DataFormat[] =
      [undefined, 'channelsFirst', 'channelsLast'];
  const paddingModes: PaddingMode[] = [undefined, 'same', 'valid'];
  const stride = 1;

  for (const outChannels of outChannelsArray) {
    for (const dataFormat of dataFormats) {
      for (const paddingMode of paddingModes) {
        const testTitle = `outChannels=${outChannels}, stride=${stride}, ` +
            `${paddingMode}, ${dataFormat}`;
        it(testTitle, () => {
          let x: Tensor = tensor3d(xLength4Data, [1, 4, 1]);
          if (dataFormat === 'channelsFirst') {
            x = K.transpose(x, [0, 2, 1]);  // NWC -> NCW.
          }

          let kernelData: number[] = [];
          let biasData: number[] = [];
          for (let i = 0; i < outChannels; ++i) {
            kernelData = kernelData.concat(kernelLength2Data);
            biasData = biasData.concat([biasScalarData + i]);
          }
          const kernel =
              K.transpose(tensor3d(kernelData, [1, outChannels, 2]), [2, 0, 1]);
          const bias = tensor1d(biasData);

          const y = K.conv1dWithBias(
              x, kernel, bias, stride, paddingMode, dataFormat);

          let yExpectedShape: [number, number, number];
          let yExpectedData: number[];
          if (paddingMode === 'valid' || paddingMode === undefined) {
            if (outChannels === 1) {
              yExpectedShape = [1, 3, 1];
              yExpectedData = [-7.8, -17.8, -37.8];
            } else if (outChannels === 2) {
              yExpectedShape = [1, 3, 2];
              yExpectedData = [-7.8, -6.8, -17.8, -16.8, -37.8, -36.8];
            }
          } else if (paddingMode === 'same') {
            if (outChannels === 1) {
              yExpectedShape = [1, 4, 1];
              yExpectedData = [-7.8, -17.8, -37.8, 82.2];
            } else if (outChannels === 2) {
              yExpectedShape = [1, 4, 2];
              yExpectedData =
                  [-7.8, -6.8, -17.8, -16.8, -37.8, -36.8, 82.2, 83.2];
            }
          }
          expectTensorsClose(y, tensor3d(yExpectedData, yExpectedShape));
        });
      }
    }
  }
});

describeMathCPUAndGPU('conv1d', () => {
  const xLength4Data = [10, 20, 40, 80];
  const kernelLength2Data = [1, -1];

  const stride = 2;
  const outChannels = 2;
  const dataFormat = 'channelsLast';
  const paddingMode = 'valid';
  const testTitle = `outChannels=${outChannels}, stride=${stride}, ` +
      `${paddingMode}, ${dataFormat}`;
  it(testTitle, () => {
    const x = tensor3d(xLength4Data, [1, 4, 1]);
    let kernelData: number[] = [];
    for (let i = 0; i < outChannels; ++i) {
      kernelData = kernelData.concat(kernelLength2Data);
    }
    const kernel =
        K.transpose(tensor3d(kernelData, [1, outChannels, 2]), [2, 0, 1]);
    const y = K.conv1d(x, kernel, stride, paddingMode, dataFormat);
    expectTensorsClose(y, tensor3d([-10, -10, -40, -40], [1, 2, 2]));
  });
});

describeMathCPUAndGPU('conv2d', () => {
  const x4by4Data = [[[
    [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
    [-20, -40, -60, -80]
  ]]];
  const kernel2by2Data = [1, 0, 0, -1];

  const dataFormats: DataFormat[] =
      [undefined, 'channelsFirst', 'channelsLast'];
  const paddingModes: PaddingMode[] = [undefined, 'same', 'valid'];
  const stridesArray = [1, 2];

  for (const dataFormat of dataFormats) {
    for (const paddingMode of paddingModes) {
      for (const stride of stridesArray) {
        const testTitle = `stride=${stride}, ${paddingMode}, ` +
            `${dataFormat}`;
        it(testTitle, () => {
          let x: Tensor = tensor4d(x4by4Data, [1, 1, 4, 4]);
          if (dataFormat !== 'channelsFirst') {
            x = K.transpose(x, [0, 2, 3, 1]);  // NCHW -> NHWC.
          }
          const kernel = tensor4d(kernel2by2Data, [2, 2, 1, 1]);
          const y = K.conv2d(x, kernel, [stride, stride], 'valid', dataFormat);

          let yExpected: Tensor;
          if (stride === 1) {
            yExpected = tensor4d(
                [[[[-30, -30, -30], [50, 90, 130], [30, 30, 30]]]],
                [1, 1, 3, 3]);
          } else if (stride === 2) {
            yExpected = tensor4d([[[[-30, -30], [30, 30]]]], [1, 1, 2, 2]);
          }
          if (dataFormat !== 'channelsFirst') {
            yExpected = K.transpose(yExpected, [0, 2, 3, 1]);
          }
          expectTensorsClose(y, yExpected);
        });
      }
    }
  }
});

describeMathCPUAndGPU('conv2dWithBias', () => {
  const x4by4Data = [[[
    [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
    [-20, -40, -60, -80]
  ]]];
  const kernel2by2Data = [1, 0, 0, -1];
  const biasScalarData = [2.2];

  const outChannelsArray = [2, 3];
  const dataFormats: DataFormat[] =
      [undefined, 'channelsFirst', 'channelsLast'];
  const paddingModes: PaddingMode[] = [undefined, 'same', 'valid'];
  const stridesArray = [1, 2];

  for (const outChannels of outChannelsArray) {
    for (const dataFormat of dataFormats) {
      for (const paddingMode of paddingModes) {
        for (const stride of stridesArray) {
          const testTitle = `outChannels=${outChannels}, stride=${stride}, ` +
              `${paddingMode}, ${dataFormat}`;
          it(testTitle, () => {
            let x: Tensor = tensor4d(x4by4Data, [1, 1, 4, 4]);
            if (dataFormat !== 'channelsFirst') {
              x = K.transpose(x, [0, 2, 3, 1]);  // NCHW -> NHWC.
            }

            let kernelData: number[] = [];
            let biasData: number[] = [];
            for (let i = 0; i < outChannels; ++i) {
              kernelData = kernelData.concat(kernel2by2Data);
              biasData = biasData.concat(biasScalarData);
            }
            const kernel = K.transpose(
                tensor4d(kernelData, [outChannels, 2, 2, 1]), [1, 2, 3, 0]);
            const bias = tensor1d(biasData);

            const y = K.conv2dWithBias(
                x, kernel, bias, [stride, stride], 'valid', dataFormat);

            let yExpectedShape: [number, number, number, number];
            let yExpectedDataPerChannel: number[];
            if (stride === 1) {
              yExpectedShape = [1, outChannels, 3, 3];
              yExpectedDataPerChannel =
                  [-30, -30, -30, 50, 90, 130, 30, 30, 30];
            } else if (stride === 2) {
              yExpectedShape = [1, outChannels, 2, 2];
              yExpectedDataPerChannel = [-30, -30, 30, 30];
            }
            for (let i = 0; i < yExpectedDataPerChannel.length; ++i) {
              yExpectedDataPerChannel[i] += biasScalarData[0];
            }
            let yExpectedData: number[] = [];
            for (let i = 0; i < outChannels; ++i) {
              yExpectedData = yExpectedData.concat(yExpectedDataPerChannel);
            }
            let yExpected: Tensor = tensor4d(yExpectedData, yExpectedShape);
            if (dataFormat !== 'channelsFirst') {
              yExpected = K.transpose(yExpected, [0, 2, 3, 1]);
            }
            expectTensorsClose(y, yExpected);
          });
        }
      }
    }
  }
});

describeMathCPUAndGPU('depthwiseConv2d', () => {
  const x4by4Data = [[[
    [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
    [-20, -40, -60, -80]
  ]]];

  const dataFormats: DataFormat[] =
      [undefined, 'channelsFirst', 'channelsLast'];
  const paddingModes: PaddingMode[] = [undefined, 'same', 'valid'];
  const stridesArray = [1, 2];
  const depthMultipliers = [1, 2];

  for (const dataFormat of dataFormats) {
    for (const paddingMode of paddingModes) {
      for (const stride of stridesArray) {
        for (const depthMultiplier of depthMultipliers) {
          const testTitle = `stride=${stride}, ${paddingMode}, ` +
              `${dataFormat}, depthMultiplier=${depthMultiplier}`;
          it(testTitle, () => {
            let x: Tensor = tensor4d(x4by4Data, [1, 1, 4, 4]);
            if (dataFormat !== 'channelsFirst') {
              x = K.transpose(x, [0, 2, 3, 1]);  // NCHW -> NHWC.
            }

            let kernel: Tensor4D;
            if (depthMultiplier === 1) {
              kernel = tensor4d([1, 0, 0, -1], [2, 2, 1, 1]);
            } else if (depthMultiplier === 2) {
              // Two kernels of the same absolute values but opposite signs:
              //   [[1, 0], [0, -1]] and [[-1, 0], [0, 1]].
              kernel = tensor4d([1, -1, 0, 0, 0, 0, -1, 1], [2, 2, 1, 2]);
            }
            const y = K.depthwiseConv2d(
                x, kernel, [stride, stride], 'valid', dataFormat);

            let yExpected: Tensor;
            if (stride === 1) {
              if (depthMultiplier === 1) {
                yExpected = tensor4d(
                    [[[[-30, -30, -30], [50, 90, 130], [30, 30, 30]]]],
                    [1, 1, 3, 3]);
              } else if (depthMultiplier === 2) {
                yExpected = tensor4d(
                    [[
                      [[-30, -30, -30], [50, 90, 130], [30, 30, 30]],
                      [[30, 30, 30], [-50, -90, -130], [-30, -30, -30]]
                    ]],
                    [1, 2, 3, 3]);
              }
            } else if (stride === 2) {
              if (depthMultiplier === 1) {
                yExpected = tensor4d([[[[-30, -30], [30, 30]]]], [1, 1, 2, 2]);
              } else if (depthMultiplier === 2) {
                yExpected = tensor4d(
                    [[[[-30, -30], [30, 30]], [[30, 30], [-30, -30]]]],
                    [1, 2, 2, 2]);
              }
            }
            if (dataFormat !== 'channelsFirst') {
              yExpected = K.transpose(yExpected, [0, 2, 3, 1]);
            }
            expectTensorsClose(y, yExpected);
          });
        }
      }
    }
  }

  it('Non-4D kernel leads to exception', () => {
    const x = K.zeros([1, 1, 4, 4]);
    expect(() => K.depthwiseConv2d(x, K.zeros([1, 2, 2]), [
      1, 1
    ])).toThrowError(/.* is required to be 4-D, but is instead 3-D/);
  });
});

describeMathCPUAndGPU('pool2d', () => {
  const x4by4Data = [[[
    [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
    [-20, -40, -60, -80]
  ]]];
  const x5by5Data = [[[
    [0, 1, 3, 5, 7], [0, 2, 4, 6, 8], [0, 0, 0, 0, 0], [0, -1, -3, -5, -7],
    [0, -2, -4, -6, -8]
  ]]];

  const poolModes: PoolMode[] = [undefined, 'max', 'avg'];
  const dataFormats: DataFormat[] =
      [undefined, 'channelsFirst', 'channelsLast'];
  const stridesArray = [1, 2];
  for (const poolMode of poolModes) {
    for (const dataFormat of dataFormats) {
      for (const stride of stridesArray) {
        const testTitle = `4x4, ${stride}, same, ${dataFormat}, ` +
            `${poolMode}`;
        it(testTitle, () => {
          let x: Tensor = tensor4d(x4by4Data, [1, 1, 4, 4]);
          if (dataFormat !== 'channelsFirst') {
            x = K.transpose(x, [0, 2, 3, 1]);  // NCHW -> NHWC.
          }
          let yExpected: Tensor;
          if (poolMode === 'avg') {
            if (stride === 1) {
              yExpected = tensor4d(
                  [[[
                    [25, 45, 65, 37.5], [5, 5, 5, 2.5], [-25, -45, -65, -37.5],
                    [-15, -25, -35, -20]
                  ]]],
                  [1, 1, 4, 4]);
            } else {
              yExpected = tensor4d([[[[25, 65], [-25, -65]]]], [1, 1, 2, 2]);
            }
          } else {
            if (stride === 1) {
              yExpected = tensor4d(
                  [[[
                    [40, 60, 80, 80], [40, 60, 80, 80], [-10, -30, -50, -70],
                    [-20, -40, -60, -80]
                  ]]],
                  [1, 1, 4, 4]);
            } else if (stride === 2) {
              yExpected = tensor4d([[[[40, 80], [-10, -50]]]], [1, 1, 2, 2]);
            }
          }
          if (dataFormat !== 'channelsFirst') {
            yExpected = K.transpose(yExpected, [0, 2, 3, 1]);
          }
          const y = K.pool2d(
              x, [2, 2], [stride, stride], 'same', dataFormat, poolMode);
          expectTensorsClose(y, yExpected);
        });
      }
    }
  }

  for (const poolMode of poolModes) {
    it(`5x5, 2, same, CHANNEL_FIRST, ${poolMode}`, () => {
      const x5by5 = tensor4d(x5by5Data, [1, 1, 5, 5]);
      let yExpected = tensor4d(x4by4Data, [1, 1, 4, 4]);
      if (poolMode === 'avg') {
        yExpected = tensor4d(
            [[[[0.75, 4.5, 3.75], [-0.25, -2, -1.75], [-0.5, -2.5, -2]]]],
            [1, 1, 3, 3]);
      } else {
        yExpected =
            tensor4d([[[[2, 6, 8], [0, 0, 0], [0, -4, -8]]]], [1, 1, 3, 3]);
      }
      const y =
          K.pool2d(x5by5, [2, 2], [2, 2], 'same', 'channelsFirst', poolMode);
      expectTensorsClose(y, yExpected);
    });
  }

  for (const poolMode of poolModes) {
    it(`5x5, 2, valid, CHANNEL_LAST, ${poolMode}`, () => {
      const x5by5 =
          K.transpose(tensor4d(x5by5Data, [1, 1, 5, 5]), [0, 2, 3, 1]);
      let yExpected: Tensor4D;
      if (poolMode === 'avg') {
        yExpected = tensor4d([[[[0.75, 4.5], [-0.25, -2]]]], [1, 1, 2, 2]);
      } else {
        yExpected = tensor4d([[[[2, 6], [0, 0]]]], [1, 1, 2, 2]);
      }
      const y =
          K.pool2d(x5by5, [2, 2], [2, 2], 'valid', 'channelsLast', poolMode);
      expectTensorsClose(y, K.transpose(yExpected, [0, 2, 3, 1]));
    });
  }
});

describe('floatx ', () => {
  it('returns "float32"', () => {
    expect(K.floatx()).toEqual(DType.float32);
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

describeMathCPUAndGPU('Softmax ', () => {
  it('1D', () => {
    const initVals = new Float32Array([0, 1, 3, 9]);
    const initX = tensor1d(initVals);
    expectTensorsClose(
        K.softmax(initX), tensor1d([0.000, 0.000, 0.002, 0.997]));
  });
  it('all equal', () => {
    const initVals = new Float32Array([-1, -1, -1, -1]);
    const initX = tensor1d(initVals);
    expectTensorsClose(K.softmax(initX), tensor1d([0.25, 0.25, 0.25, 0.25]));
  });
  it('2D', () => {
    const initVals = new Float32Array([0, 1, 3, 9, 0, 1, 3, 9]);
    const initX = tensor2d(initVals, [2, 4]);
    expectTensorsClose(
        K.softmax(initX),
        tensor2d(
            [[0.000, 0.000, 0.002, 0.997], [0.000, 0.000, 0.002, 0.997]],
            [2, 4]));
  });
  it('3D', () => {
    const initVals = new Float32Array([0, 1, 3, 9, 0, 1, 3, 9]);
    const initX = tensor3d(initVals, [1, 2, 4]);
    expectTensorsClose(
        K.softmax(initX),
        tensor3d(
            [[[0.000, 0.000, 0.002, 0.997], [0.000, 0.000, 0.002, 0.997]]],
            [1, 2, 4]));
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
    const targetComplement = K.scalarPlusArray(scalar(1), K.neg(target));
    const outputComplement = K.scalarPlusArray(scalar(1), K.neg(output));
    return K.neg(K.add(
        K.multiply(target, K.log(output)),
        K.multiply(targetComplement, K.log(outputComplement))));
  }

  it('from logits', () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
    const sigmoidX = K.sigmoid(x);
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
    const targetComplement = K.scalarPlusArray(scalar(1), K.neg(target));
    const sigmoidX = K.sigmoid(x);
    const sigmoidXComplement = K.scalarPlusArray(scalar(1), K.neg(sigmoidX));
    const expected = K.add(
        K.multiply(target, K.neg(K.log(sigmoidX))),
        K.multiply(targetComplement, K.neg(K.log(sigmoidXComplement))));
    const result = K.sigmoidCrossEntropyWithLogits(target, x);
    expectTensorsClose(result, expected);
  });
});

describeMathCPUAndGPU('Sigmoid', () => {
  it('2D', () => {
    const xValues = [-5, -2, 0, 1, 2, 5];
    const x = tensor2d(xValues, [2, 3]);
    const y = K.sigmoid(x);
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

/**
 * A simplistic RNN step function for testing.
 * This step function simply
 * - calculates a reduced mean over all input elements, for each sample.
 * - adds that mean to the state tensor(s),
 * - take the negative of the 1st current state tensor and use it as the
 *   output.
 * @param inputs
 * @param states
 */
function rnnStepForTest(inputs: Tensor, states: Tensor[]): [Tensor, Tensor[]] {
  const mean = K.mean(inputs) as Scalar;
  const newStates = states.map(state => K.scalarPlusArray(mean, state));
  const output = K.neg(newStates[0]);
  return [output, newStates];
}

describeMathCPUAndGPU('rnn', () => {
  it('Simple step function: 3D inputs, 1 state', () => {
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    const initialStates = [K.zeros([2, 4])];
    const rnnOutputs = K.rnn(rnnStepForTest, inputs, initialStates);
    const lastOutput = rnnOutputs[0];
    const outputs = rnnOutputs[1];
    const newStates = rnnOutputs[2];
    expectTensorsClose(
        lastOutput,
        tensor2d(
            [
              [-57.75, -57.75, -57.75, -57.75],
              [-57.75, -57.75, -57.75, -57.75]
            ],
            [2, 4]));
    expectTensorsClose(
        outputs,
        tensor3d(
            [
              [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
              ],
              [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
              ]
            ],
            [2, 3, 4]));
    expect(newStates.length).toEqual(1);
    expectTensorsClose(
        newStates[0],
        tensor2d(
            [[57.75, 57.75, 57.75, 57.75], [57.75, 57.75, 57.75, 57.75]],
            [2, 4]));
  });

  it('Simple step function: 3D inputs, 2 states', () => {
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    // The two state tensors have different shapes.
    const initialStates = [K.zeros([2, 4]), K.ones([2, 3])];
    const rnnOutputs = K.rnn(rnnStepForTest, inputs, initialStates);
    const lastOutput = rnnOutputs[0];
    const outputs = rnnOutputs[1];
    const newStates = rnnOutputs[2];
    expectTensorsClose(
        lastOutput,
        tensor2d(
            [
              [-57.75, -57.75, -57.75, -57.75],
              [-57.75, -57.75, -57.75, -57.75]
            ],
            [2, 4]));
    expectTensorsClose(
        outputs,
        tensor3d(
            [
              [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
              ],
              [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
              ]
            ],
            [2, 3, 4]));
    expect(newStates.length).toEqual(2);
    expectTensorsClose(
        newStates[0],
        tensor2d(
            [[57.75, 57.75, 57.75, 57.75], [57.75, 57.75, 57.75, 57.75]],
            [2, 4]));
    expectTensorsClose(
        newStates[1],
        tensor2d([[58.75, 58.75, 58.75], [58.75, 58.75, 58.75]], [2, 3]));
  });

  it('Simple step function: 4D inputs, 2 states', () => {
    const inputs = tensor4d(
        [
          [[[1], [2]], [[3], [4]], [[5], [6]]],
          [[[10], [20]], [[30], [40]], [[50], [60]]]
        ],
        [2, 3, 2, 1]);
    // The two state tensors have different shapes.
    const initialStates = [K.zeros([2, 4]), K.ones([2, 3])];
    const rnnOutputs = K.rnn(rnnStepForTest, inputs, initialStates);
    const lastOutput = rnnOutputs[0];
    const outputs = rnnOutputs[1];
    const newStates = rnnOutputs[2];
    expectTensorsClose(
        lastOutput,
        tensor2d(
            [
              [-57.75, -57.75, -57.75, -57.75],
              [-57.75, -57.75, -57.75, -57.75]
            ],
            [2, 4]));
    expectTensorsClose(
        outputs,
        tensor3d(
            [
              [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
              ],
              [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
              ]
            ],
            [2, 3, 4]));
    expect(newStates.length).toEqual(2);
    expectTensorsClose(
        newStates[0],
        tensor2d(
            [[57.75, 57.75, 57.75, 57.75], [57.75, 57.75, 57.75, 57.75]],
            [2, 4]));
    expectTensorsClose(
        newStates[1],
        tensor2d([[58.75, 58.75, 58.75], [58.75, 58.75, 58.75]], [2, 3]));
  });

  it('Using inputs <3D leads to ValueError', () => {
    const inputs = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const initialStates = [K.zeros([4]), K.ones([3])];
    expect(() => K.rnn(rnnStepForTest, inputs, initialStates)).toThrowError();
  });
});

describeMathCPUAndGPU('gradients', () => {
  it('Simple mean: 1 variable', () => {
    const var1 =
        new LayerVariable(K.scalarTimesArray(scalar(2.0), K.ones([2, 2])));
    const gradients = K.gradients(() => K.mean(var1.read()) as Scalar, [var1]);
    expect(gradients.length).toEqual(1);
    expectTensorsClose(
        tensor2d([[0.25, 0.25], [0.25, 0.25]], [2, 2]), gradients[0]);
  });
  it('Simple matmul and mean: 2 variables', () => {
    const var1 = new LayerVariable(tensor2d([[1, 0], [0, 0]], [2, 2]));
    const var2 = new LayerVariable(tensor2d([[1, 0], [0, 1]], [2, 2]));
    const gradients = K.gradients(
        () => K.mean(K.dot(var1.read(), var2.read())) as Scalar, [var1, var2]);
    expect(gradients.length).toEqual(2);
    // d(loss) / d(var1).
    expectTensorsClose(
        tensor2d([[0.25, 0.25], [0.25, 0.25]], [2, 2]), gradients[0]);
    // d(loss) / d(var2).
    expectTensorsClose(tensor2d([[0.25, 0.25], [0, 0]], [2, 2]), gradients[1]);
  });
});
