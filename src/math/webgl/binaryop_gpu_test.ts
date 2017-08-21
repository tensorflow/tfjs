/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as test_util from '../../test_util';
// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, Array3D, initializeGPU, NDArray, Scalar} from '../ndarray';

import {BinaryOpProgram} from './binaryop_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {TextureManager} from './texture_manager';

describe('binaryop_gpu Add', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Scalar.new(0);
    const b = Array2D.zeros([12, 513]);
    const result = uploadBinaryOpDownload(a, b, '+');
    expect(result.length).toEqual(b.size);
  });

  it('preserves the matrix when the scalar is 0', () => {
    const c = Scalar.new(0);
    const a = Array1D.new([1, 2, 3]);
    const result = uploadBinaryOpDownload(c, a, '+');
    test_util.expectArraysClose(result, new Float32Array([1, 2, 3]), 0);
  });

  it('adds the scalar to every element in the matrix', () => {
    const a = Array1D.new([1, 2, 3, 4]);
    const c = Scalar.new(0.5);
    const result = uploadBinaryOpDownload(c, a, '+');
    test_util.expectArraysClose(
        result, new Float32Array([1.5, 2.5, 3.5, 4.5]), 0.0001);
  });
});

describe('binaryop_gpu Sub', () => {
  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array2D.zeros([12, 513]);
    const c = Scalar.new(0);
    const result = uploadBinaryOpDownload(a, c, '-');
    expect(result.length).toEqual(a.size);
  });

  it('preserves the matrix when the scalar is 0', () => {
    const a = Array1D.new([1, 2, 3]);
    const c = Scalar.new(0);
    const result = uploadBinaryOpDownload(a, c, '-');
    test_util.expectArraysClose(result, new Float32Array([1, 2, 3]), 0);
  });

  it('subtracts the scalar from every element in the matrix', () => {
    const a = Array1D.new([1, 2, 3, 4]);
    const c = Scalar.new(0.5);
    const result = uploadBinaryOpDownload(a, c, '-');
    test_util.expectArraysClose(
        result, new Float32Array([0.5, 1.5, 2.5, 3.5]), 0.0001);
  });

  it('2D - 1D broadcasting', () => {
    const a = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]]);
    const b = Array1D.new([1, 3]);
    const result = uploadBinaryOpDownload(a, b, '-');
    test_util.expectArraysClose(
        result, new Float32Array([0, -1, 2, 1, 4, 3]), 1e-4);
  });

  it('2D - 1D invalid shapes for broadcasting', () => {
    const a = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]]);
    const b = Array1D.new([1, 2, 3]);
    // shape [3, 2] is not compatible with shape [3].
    const f = () => uploadBinaryOpDownload(a, b, '-');
    expect(f).toThrowError();
  });

  it('3D - 2D broadcasting', () => {
    const a = Array3D.new([2, 2, 2], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const b = Array2D.new([2, 2], [[1, 2], [3, 5]]);
    // shape [3, 2] is not compatible with shape [3].
    const res = uploadBinaryOpDownload(a, b, '-');
    test_util.expectArraysClose(
        res, new Float32Array([0, 0, 0, -1, 4, 4, 4, 3]), 1e-4);
  });
});


describe('binaryop_gpu Mul', () => {
  function cpuMultiply(a: Float32Array, b: Float32Array): Float32Array {
    const result = new Float32Array(a.length);
    for (let i = 0; i < result.length; ++i) {
      result[i] = a[i] * b[i];
    }
    return result;
  }

  it('returns a matrix with the same shape as the input matrix', () => {
    const a = Array2D.zeros([12, 513]);
    const c = Scalar.new(0);
    const result = uploadBinaryOpDownload(c, a, '*');
    expect(result.length).toEqual(a.size);
  });

  it('zeros out the matrix when the scalar is 0', () => {
    const a = Array1D.new([1, 2, 3]);
    const c = Scalar.new(0);
    const result = uploadBinaryOpDownload(c, a, '*');
    test_util.expectArraysClose(result, new Float32Array([0, 0, 0]), 0);
  });

  it('triples the matrix when the scalar is 3', () => {
    const a = Array1D.new([1, 2, 3]);
    const c = Scalar.new(3);
    const result = uploadBinaryOpDownload(c, a, '*');
    test_util.expectArraysClose(result, new Float32Array([3, 6, 9]), 0);
  });

  it('sets all result entries to 0 if A is 0', () => {
    const a = Array2D.zeros([25, 25]);
    const expected = a.getValues();
    const b = Array2D.zerosLike(a);
    b.fill(1.0);
    const result = uploadBinaryOpDownload(a, b, '*');
    expect(result).toEqual(expected);
  });

  it('sets all result entries to 0 if B is 0', () => {
    const a = Array2D.zeros([25, 25]);
    a.fill(1.0);
    const b = Array2D.zerosLike(a);
    const expected = b.getValues();
    const result = uploadBinaryOpDownload(a, b, '*');
    expect(result).toEqual(expected);
  });

  it('sets all result entries to A if B is [1]', () => {
    const a = Array1D.new(test_util.randomArrayInRange(16, -10, 10));
    const expected = a.getValues();
    const b = Array1D.zeros([16]);
    b.fill(1.0);
    const result = uploadBinaryOpDownload(a, b, '*');
    test_util.expectArraysClose(result, expected, 0.0001);
  });

  it('writes the element-wise product of A and B to result', () => {
    const a = Array1D.new(test_util.randomArrayInRange(64, -10, 10));
    const b = Array1D.new(test_util.randomArrayInRange(64, -10, 10));
    const expected = cpuMultiply(a.getValues(), b.getValues());
    const result = uploadBinaryOpDownload(a, b, '*');
    test_util.expectArraysClose(result, expected, 0.0001);
  });
});

describe('binaryop_gpu Divide', () => {
  it('Scalar / Matrix', () => {
    const c = Scalar.new(2);
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const r = uploadBinaryOpDownload(c, a, '/');
    expect(r[0]).toBeCloseTo(2 / 1);
    expect(r[1]).toBeCloseTo(2 / 2);
    expect(r[2]).toBeCloseTo(2 / 3);
    expect(r[3]).toBeCloseTo(2 / 4);
    expect(r[4]).toBeCloseTo(2 / 5);
    expect(r[5]).toBeCloseTo(2 / 6);
  });
});

function uploadBinaryOpDownload(
    a: NDArray, b: NDArray, op: '+'|'-'|'*'|'/'): Float32Array {
  const gpgpu = new GPGPUContext();
  const textureManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, textureManager);

  const program = new BinaryOpProgram(op, a.shape, b.shape);
  const res = NDArray.zeros(program.outputShape);
  const binary = gpgpu_math.compileProgram(gpgpu, program, [a, b], res);
  gpgpu_math.runProgram(binary, [a, b], res);

  const resValues = res.getValues();
  textureManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();

  return resValues;
}
