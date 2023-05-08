/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {backend_util, BinaryInputs, DataType, KernelFunc, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {cast} from '../kernels/Cast';
import {complex} from '../kernels/Complex';

import {ComplexBinaryKernelImpl, ComplexBinaryOperation, SimpleBinaryKernelImpl} from './binary_types';

/**
 * Template that creates a `KernelFunc` for binary ops.
 * @param name Kernel name.
 * @param binaryKernelImpl A `SimpleBinaryKernelImpl` for the kernel.
 * @param binaryKernelComplexImpl Optional. If exists, represents a
 *     `ComplexBinaryKernelImpl` for the kernel, will be used when input dtype
 *     is `complex64`.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
export function binaryKernelFunc(
    name: string, simpleImpl: SimpleBinaryKernelImpl,
    complexImpl?: ComplexBinaryKernelImpl, dtype?: DataType): KernelFunc {
  if (complexImpl == null) {
    return ({inputs, backend}) => {
      const {a, b} = inputs as BinaryInputs;
      const cpuBackend = backend as MathBackendCPU;

      assertNotComplex([a, b], name);

      const aVals = cpuBackend.data.get(a.dataId).values as TypedArray;
      const bVals = cpuBackend.data.get(b.dataId).values as TypedArray;

      const decodedAVals = a.dtype === 'string' ?
          // tslint:disable-next-line: no-any
          backend_util.fromUint8ToStringArray(aVals as any as Uint8Array[]) :
          aVals;
      const decodedBVals = a.dtype === 'string' ?
          // tslint:disable-next-line: no-any
          backend_util.fromUint8ToStringArray(bVals as any as Uint8Array[]) :
          bVals;
      const $dtype = dtype || a.dtype;

      const [resultData, resultShape] =
          simpleImpl(a.shape, b.shape, decodedAVals, decodedBVals, $dtype);

      return cpuBackend.makeTensorInfo(resultShape, $dtype, resultData);
    };
  }

  return ({inputs, backend}) => {
    const {a, b} = inputs as BinaryInputs;
    const cpuBackend = backend as MathBackendCPU;

    if (a.dtype === 'complex64' || b.dtype === 'complex64') {
      const $aComplex = cast(
          {inputs: {x: a}, backend: cpuBackend, attrs: {dtype: 'complex64'}});

      const $aComplexVals = cpuBackend.data.get($aComplex.dataId);

      const aReal = $aComplexVals.complexTensorInfos.real;
      const aImag = $aComplexVals.complexTensorInfos.imag;

      const aRealVals =
          cpuBackend.data.get(aReal.dataId).values as Float32Array;
      const aImagVals =
          cpuBackend.data.get(aImag.dataId).values as Float32Array;

      const $bComplex = cast(
          {inputs: {x: b}, backend: cpuBackend, attrs: {dtype: 'complex64'}});

      const $bComplexVals = cpuBackend.data.get($bComplex.dataId);

      const bReal = $bComplexVals.complexTensorInfos.real;
      const bImag = $bComplexVals.complexTensorInfos.imag;

      const bRealVals =
          cpuBackend.data.get(bReal.dataId).values as Float32Array;
      const bImagVals =
          cpuBackend.data.get(bImag.dataId).values as Float32Array;

      const [resultRealData, resultImagData, resultShape] = complexImpl(
          a.shape, b.shape, aRealVals, aImagVals, bRealVals, bImagVals);

      const resultReal =
          cpuBackend.makeTensorInfo(resultShape, 'float32', resultRealData);

      const resultImag =
          cpuBackend.makeTensorInfo(resultShape, 'float32', resultImagData);

      const result = complex(
          {inputs: {real: resultReal, imag: resultImag}, backend: cpuBackend});

      cpuBackend.disposeIntermediateTensorInfo($aComplex);
      cpuBackend.disposeIntermediateTensorInfo($bComplex);
      cpuBackend.disposeIntermediateTensorInfo(resultReal);
      cpuBackend.disposeIntermediateTensorInfo(resultImag);

      return result;
    } else {
      const aVals = cpuBackend.data.get(a.dataId).values as TypedArray;
      const bVals = cpuBackend.data.get(b.dataId).values as TypedArray;

      const $dtype = dtype || a.dtype;

      const [resultData, resultShape] =
          simpleImpl(a.shape, b.shape, aVals, bVals, $dtype);

      return cpuBackend.makeTensorInfo(resultShape, $dtype, resultData);
    }
  };
}

/**
 * Template that creates the complex type implementation for binary ops.
 * Supports broadcast.
 */
export function createComplexBinaryKernelImpl(op: ComplexBinaryOperation):
    ComplexBinaryKernelImpl {
  return (aShape: number[], bShape: number[], aRealVals: Float32Array,
          aImagVals: Float32Array, bRealVals: Float32Array,
          bImagVals: Float32Array): [TypedArray, TypedArray, number[]] => {
    const resultShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    const resultSize = util.sizeFromShape(resultShape);
    const resultRank = resultShape.length;
    const resultStrides = util.computeStrides(resultShape);

    const resultRealVals = util.getTypedArrayFromDType('float32', resultSize);
    const resultImagVals = util.getTypedArrayFromDType('float32', resultSize);

    const aBroadcastDims = backend_util.getBroadcastDims(aShape, resultShape);
    const bBroadcastDims = backend_util.getBroadcastDims(bShape, resultShape);

    const aVals = backend_util.mergeRealAndImagArrays(aRealVals, aImagVals);
    const bVals = backend_util.mergeRealAndImagArrays(bRealVals, bImagVals);

    const aRank = aShape.length;
    const aStrides = util.computeStrides(aShape);

    const bRank = bShape.length;
    const bStrides = util.computeStrides(bShape);

    if (aBroadcastDims.length + bBroadcastDims.length === 0) {
      for (let i = 0; i < resultRealVals.length; i++) {
        const aIdx = i % aVals.length;
        const bIdx = i % bVals.length;

        const result =
            op(aVals[aIdx * 2], aVals[aIdx * 2 + 1], bVals[bIdx * 2],
               bVals[bIdx * 2 + 1]);

        resultRealVals[i] = result.real;
        resultImagVals[i] = result.imag;
      }
    } else {
      for (let i = 0; i < resultRealVals.length; i++) {
        const loc = util.indexToLoc(i, resultRank, resultStrides);

        const aLoc = loc.slice(-aRank);
        aBroadcastDims.forEach(d => aLoc[d] = 0);
        const aIndex = util.locToIndex(aLoc, aRank, aStrides);

        const bLoc = loc.slice(-bRank);
        bBroadcastDims.forEach(d => bLoc[d] = 0);
        const bIndex = util.locToIndex(bLoc, bRank, bStrides);

        const opResult =
            op(aVals[aIndex * 2], aVals[aIndex * 2 + 1], bVals[bIndex * 2],
               bVals[bIndex * 2 + 1]);

        resultRealVals[i] = opResult.real;
        resultImagVals[i] = opResult.imag;
      }
    }
    return [resultRealVals, resultImagVals, resultShape];
  };
}
