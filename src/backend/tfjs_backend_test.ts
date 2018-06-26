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
import {DataType, memory, scalar, tensor1d, tensor2d, tensor3d, tensor4d, zeros} from '@tensorflow/tfjs-core';

import {SymbolicTensor} from '../engine/topology';
import {range} from '../utils/math_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectNoLeakedTensors, expectTensorsClose} from '../utils/test_utils';
import {LayerVariable} from '../variables';

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

describe('dtype', () => {
  it('returns float32 for an Tensor', () => {
    const x = zeros([1]);
    expect(x.dtype).toEqual('float32');
  });

  it('returns float32 for a SymbolicTensor', () => {
    const x = new SymbolicTensor('float32', [1], null, [], {});
    expect(x.dtype).toEqual('float32');
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

describeMathCPUAndGPU('randomNormal', () => {
  const dtypes: DataType[] = ['float32', 'int32'];
  for (const dtype of dtypes) {
    // TODO(bileschi): Add probabilistic assertions on values here.
    it(`Scalar ${dtype}`, () => {
      const s = K.randomNormal([], 0, 10, dtype as 'float32' | 'int32');
      expect(s.shape).toEqual([]);
    });

    it(`1D ${dtype}`, () => {
      const v = K.randomNormal([20], 0, 2, dtype as 'float32' | 'int32');
      expect(v.shape).toEqual([20]);
    });

    it(`2D ${dtype}`, () => {
      const x = K.randomNormal([3, 20], -10, 20, dtype as 'float32' | 'int32');
      expect(x.shape).toEqual([3, 20]);
    });

    it(`3D ${dtype}`, () => {
      const y =
          K.randomNormal([2, 3, 4], 100, 10, dtype as 'float32' | 'int32');
      expect(y.shape).toEqual([2, 3, 4]);
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
  it('Does not leak', () => {
    const input = tensor2d([-1, 0, 1, -1], [2, 2]);
    expectNoLeakedTensors(() => K.softsign(input), 1);
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
