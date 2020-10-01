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

import {backend_util, Tensor, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {add} from '../kernels/Add';
import {complex} from '../kernels/Complex';
import {concat} from '../kernels/Concat';
import {divConfig} from '../kernels/Div';
import {identity} from '../kernels/Identity';
import {imag} from '../kernels/Imag';
import {multiply} from '../kernels/Multiply';
import {real} from '../kernels/Real';
import {slice} from '../kernels/Slice';
import {sub} from '../kernels/Sub';

/**
 * Calculate FFT of inner most elements of batch tensor.
 */
export function fftBatch(
    input: TensorInfo, inverse: boolean,
    cpuBackend: MathBackendCPU): TensorInfo {
  const inputShape = input.shape;
  const batch = inputShape[0];
  const innerDim = inputShape[1];

  const inputVals = cpuBackend.data.get(input.dataId);

  const real2D = inputVals.complexTensorInfos.real;
  const imag2D = inputVals.complexTensorInfos.imag;

  // Collects real and imaginary values separately.
  const resultShape = [batch, innerDim];
  const resultSize = util.sizeFromShape(resultShape);
  const resultReal = util.getTypedArrayFromDType('float32', resultSize);
  const resultImag = util.getTypedArrayFromDType('float32', resultSize);

  for (let b = 0; b < batch; b++) {
    // TODO: Support slice ops for complex type.
    const r = slice({
      inputs: {x: real2D},
      backend: cpuBackend,
      attrs: {begin: [b, 0], size: [1, innerDim]}
    });
    const i = slice({
      inputs: {x: imag2D},
      backend: cpuBackend,
      attrs: {begin: [b, 0], size: [1, innerDim]}
    });

    const input = complex({inputs: {real: r, imag: i}, backend: cpuBackend});

    // Run FFT by batch element.
    const {real, imag} = fftImpl(input, inverse, cpuBackend);
    const res = backend_util.mergeRealAndImagArrays(real, imag);

    for (let d = 0; d < innerDim; d++) {
      const c = backend_util.getComplexWithIndex(res, d);
      resultReal[b * innerDim + d] = c.real;
      resultImag[b * innerDim + d] = c.imag;
    }

    cpuBackend.disposeIntermediateTensorInfo(r);
    cpuBackend.disposeIntermediateTensorInfo(i);
    cpuBackend.disposeIntermediateTensorInfo(input);
  }

  const $realInfo: TensorInfo =
      cpuBackend.makeTensorInfo(resultShape, 'float32', resultReal);
  const $imagInfo: TensorInfo =
      cpuBackend.makeTensorInfo(resultShape, 'float32', resultImag);

  const result = complex(
      {inputs: {real: $realInfo, imag: $imagInfo}, backend: cpuBackend});

  cpuBackend.disposeIntermediateTensorInfo($realInfo);
  cpuBackend.disposeIntermediateTensorInfo($imagInfo);

  return result;
}

export function fftImpl(
    input: TensorInfo, inverse: boolean,
    cpuBackend: MathBackendCPU): {real: Float32Array, imag: Float32Array} {
  const inputSize = util.sizeFromShape(input.shape);

  const inputVals = cpuBackend.data.get(input.dataId);

  const realVals =
      cpuBackend.data.get(inputVals.complexTensorInfos.real.dataId).values as
      Float32Array;

  const imagVals =
      cpuBackend.data.get(inputVals.complexTensorInfos.imag.dataId).values as
      Float32Array;

  if (isExponentOf2(inputSize)) {
    const result =
        fftRadix2(realVals, imagVals, inputSize, inverse, cpuBackend);

    const resultShape = [input.shape[0], input.shape[1]];

    if (inverse) {
      const realInfo: TensorInfo =
          cpuBackend.makeTensorInfo(resultShape, 'float32', result.real);
      const imagInfo: TensorInfo =
          cpuBackend.makeTensorInfo(resultShape, 'float32', result.imag);

      const sizeInfo: TensorInfo = cpuBackend.makeTensorInfo(
          [], 'float32',
          util.createScalarValue(inputSize as {} as 'float32', 'float32'));
      const sizeInfoCopy =
          identity({inputs: {x: sizeInfo}, backend: cpuBackend});

      const divRealInfo =
          divConfig.kernelFunc(
              {inputs: {a: realInfo, b: sizeInfo}, backend: cpuBackend}) as
          TensorInfo;
      const divImagInfo =
          divConfig.kernelFunc(
              {inputs: {a: imagInfo, b: sizeInfoCopy}, backend: cpuBackend}) as
          TensorInfo;

      const divRealVals =
          cpuBackend.data.get(divRealInfo.dataId).values as Float32Array;
      const divImagVals =
          cpuBackend.data.get(divImagInfo.dataId).values as Float32Array;

      cpuBackend.disposeIntermediateTensorInfo(realInfo);
      cpuBackend.disposeIntermediateTensorInfo(imagInfo);
      cpuBackend.disposeIntermediateTensorInfo(sizeInfo);
      cpuBackend.disposeIntermediateTensorInfo(sizeInfoCopy);
      cpuBackend.disposeIntermediateTensorInfo(divRealInfo);
      cpuBackend.disposeIntermediateTensorInfo(divImagInfo);

      return {real: divRealVals, imag: divImagVals};
    }

    return result;
  } else {
    const data = backend_util.mergeRealAndImagArrays(realVals, imagVals);

    const rawOutput =
        fourierTransformByMatmul(data, inputSize, inverse) as Float32Array;

    return backend_util.splitRealAndImagArrays(rawOutput);
  }
}

function isExponentOf2(size: number): boolean {
  return (size & size - 1) === 0;
}

// FFT using Cooley-Tukey algorithm on radix 2 dimensional input.
function fftRadix2(
    realVals: Float32Array, imagVals: Float32Array, size: number,
    inverse: boolean,
    cpuBackend: MathBackendCPU): {real: Float32Array, imag: Float32Array} {
  if (size === 1) {
    return {real: realVals, imag: imagVals};
  }

  const data = backend_util.mergeRealAndImagArrays(realVals, imagVals);

  const half = size / 2;

  const evenComplex = backend_util.complexWithEvenIndex(data);

  const evenRealVals = evenComplex.real;
  const evenImagVals = evenComplex.imag;

  const evenShape = [evenRealVals.length];

  const evenRealInfo =
      cpuBackend.makeTensorInfo(evenShape, 'float32', evenRealVals);
  const evenImagInfo =
      cpuBackend.makeTensorInfo(evenShape, 'float32', evenImagVals);

  const evenTensorInfo = complex(
      {inputs: {real: evenRealInfo, imag: evenImagInfo}, backend: cpuBackend});

  const oddComplex = backend_util.complexWithOddIndex(data);

  const oddRealVals = oddComplex.real;
  const oddImagVals = oddComplex.imag;

  const oddShape = [oddRealVals.length];

  const oddRealInfo =
      cpuBackend.makeTensorInfo(oddShape, 'float32', oddRealVals);
  const oddImagInfo =
      cpuBackend.makeTensorInfo(oddShape, 'float32', oddImagVals);

  const oddTensorInfo = complex(
      {inputs: {real: oddRealInfo, imag: oddImagInfo}, backend: cpuBackend});

  // Recursive call for half part of original input.
  const $evenComplex =
      fftRadix2(evenRealVals, evenImagVals, half, inverse, cpuBackend);

  const $evenRealVals = $evenComplex.real;
  const $evenImagVals = $evenComplex.imag;

  const $evenShape = [$evenRealVals.length];

  const $evenRealInfo =
      cpuBackend.makeTensorInfo($evenShape, 'float32', $evenRealVals);
  const $evenImagInfo =
      cpuBackend.makeTensorInfo($evenShape, 'float32', $evenImagVals);

  const $evenTensorInfo = complex({
    inputs: {real: $evenRealInfo, imag: $evenImagInfo},
    backend: cpuBackend
  });

  const $oddComplex =
      fftRadix2(oddRealVals, oddImagVals, half, inverse, cpuBackend);

  const $oddRealVals = $oddComplex.real;
  const $oddImagVals = $oddComplex.imag;

  const $oddShape = [$oddRealVals.length];

  const $oddRealInfo =
      cpuBackend.makeTensorInfo($oddShape, 'float32', $oddRealVals);
  const $oddImagInfo =
      cpuBackend.makeTensorInfo($oddShape, 'float32', $oddImagVals);

  const $oddTensorInfo = complex(
      {inputs: {real: $oddRealInfo, imag: $oddImagInfo}, backend: cpuBackend});

  const e = backend_util.exponents(size, inverse);
  const eShape = [e.real.length];

  const eRealInfo = cpuBackend.makeTensorInfo(eShape, 'float32', e.real);
  const eImagInfo = cpuBackend.makeTensorInfo(eShape, 'float32', e.imag);

  const complexInfo = complex(
      {inputs: {real: eRealInfo, imag: eImagInfo}, backend: cpuBackend});

  const exponentInfo =
      multiply(
          {inputs: {a: complexInfo, b: $oddTensorInfo}, backend: cpuBackend}) as
      TensorInfo;

  const addPart = add({
                    inputs: {a: $evenTensorInfo, b: exponentInfo},
                    backend: cpuBackend
                  }) as TensorInfo;
  const subPart = sub({
                    inputs: {a: $evenTensorInfo, b: exponentInfo},
                    backend: cpuBackend
                  }) as TensorInfo;

  const addPartReal = real({inputs: {input: addPart}, backend: cpuBackend});
  const subPartReal = real({inputs: {input: subPart}, backend: cpuBackend});

  const addPartImag = imag({inputs: {input: addPart}, backend: cpuBackend});
  const subPartImag = imag({inputs: {input: subPart}, backend: cpuBackend});

  const $real = concat({
    inputs: [addPartReal as Tensor, subPartReal as Tensor],
    backend: cpuBackend,
    attrs: {axis: 0}
  });
  const $imag = concat({
    inputs: [addPartImag as Tensor, subPartImag as Tensor],
    backend: cpuBackend,
    attrs: {axis: 0}
  });

  const $realVals = cpuBackend.data.get($real.dataId).values as Float32Array;
  const $imagVals = cpuBackend.data.get($imag.dataId).values as Float32Array;

  cpuBackend.disposeIntermediateTensorInfo(evenRealInfo);
  cpuBackend.disposeIntermediateTensorInfo(evenImagInfo);
  cpuBackend.disposeIntermediateTensorInfo(evenTensorInfo);
  cpuBackend.disposeIntermediateTensorInfo(oddRealInfo);
  cpuBackend.disposeIntermediateTensorInfo(oddImagInfo);
  cpuBackend.disposeIntermediateTensorInfo(oddTensorInfo);
  cpuBackend.disposeIntermediateTensorInfo($evenRealInfo);
  cpuBackend.disposeIntermediateTensorInfo($evenImagInfo);
  cpuBackend.disposeIntermediateTensorInfo($evenTensorInfo);
  cpuBackend.disposeIntermediateTensorInfo($oddRealInfo);
  cpuBackend.disposeIntermediateTensorInfo($oddImagInfo);
  cpuBackend.disposeIntermediateTensorInfo($oddTensorInfo);
  cpuBackend.disposeIntermediateTensorInfo(eRealInfo);
  cpuBackend.disposeIntermediateTensorInfo(eImagInfo);
  cpuBackend.disposeIntermediateTensorInfo(complexInfo);
  cpuBackend.disposeIntermediateTensorInfo(exponentInfo);
  cpuBackend.disposeIntermediateTensorInfo(addPart);
  cpuBackend.disposeIntermediateTensorInfo(subPart);
  cpuBackend.disposeIntermediateTensorInfo(addPartReal);
  cpuBackend.disposeIntermediateTensorInfo(addPartImag);
  cpuBackend.disposeIntermediateTensorInfo(subPartReal);
  cpuBackend.disposeIntermediateTensorInfo(subPartImag);
  cpuBackend.disposeIntermediateTensorInfo($real);
  cpuBackend.disposeIntermediateTensorInfo($imag);

  return {real: $realVals, imag: $imagVals};
}

// Calculate fourier transform by multplying sinusoid matrix.
function fourierTransformByMatmul(
    data: TypedArray, size: number, inverse: boolean): TypedArray {
  const ret = new Float32Array(size * 2);
  // TODO: Use matmul instead once it supports complex64 type.
  for (let r = 0; r < size; r++) {
    let real = 0.0;
    let imag = 0.0;
    for (let c = 0; c < size; c++) {
      const e = backend_util.exponent(r * c, size, inverse);
      const term = backend_util.getComplexWithIndex(data as Float32Array, c);
      real += term.real * e.real - term.imag * e.imag;
      imag += term.real * e.imag + term.imag * e.real;
    }
    if (inverse) {
      real /= size;
      imag /= size;
    }
    backend_util.assignToTypedArray(ret, real, imag, r);
  }
  return ret;
}
