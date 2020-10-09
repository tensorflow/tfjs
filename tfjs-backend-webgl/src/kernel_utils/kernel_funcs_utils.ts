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

import {BinaryInputs, DataType, env, KernelFunc, TypedArray, UnaryInputs, upcastType} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';
import {complex} from '../kernels/Complex';
import {UnaryOpProgram} from '../unaryop_gpu';

import {SimpleBinaryKernelImplCPU} from './shared';

export const CHECK_NAN_SNIPPET_UNARY = `if (isnan(x)) return x;`;

export const CHECK_NAN_SNIPPET_BINARY = `
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`;

export const CHECK_NAN_SNIPPET_BINARY_PACKED = `
  result.r = isNaN.r > 0. ? NAN : result.r;
  result.g = isNaN.g > 0. ? NAN : result.g;
  result.b = isNaN.b > 0. ? NAN : result.b;
  result.a = isNaN.a > 0. ? NAN : result.a;
`;

/**
 * Template that creates a `KernelFunc` for unary ops.
 * @param opSnippets Op snippet to create `UnaryOpProgram`.
 */
export function unaryKernelFunc(opSnippet: string): KernelFunc {
  return ({inputs, backend}) => {
    const {x} = inputs as UnaryInputs;
    const webglBackend = backend as MathBackendWebGL;
    const program = new UnaryOpProgram(x.shape, opSnippet);
    return webglBackend.runWebGLProgram(program, [x], x.dtype);
  };
}

type BinaryKernelFuncConfig = {
  opSnippet: string,
  packedOpSnippet?: string,
  checkOutOfBounds?: boolean,
  supportsComplex?: boolean,
  cpuKernelImpl?: SimpleBinaryKernelImplCPU,
  dtype?: DataType
};

/**
 * Template that creates a `KernelFunc` for binary ops.
 * @param opSnippet Op snippet to create `BinaryOpProgram`.
 * @param packedOpSnippet Op snippet to create `BinaryOpPackedProgram`.
 * @param checkOutOfBoundsForPackedProgram Whether to set checkOutOfBounds=true
 *     when creating BinaryOpPackedProgram.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
export function binaryKernelFunc({
  opSnippet,
  packedOpSnippet,
  checkOutOfBounds = false,
  supportsComplex = false,
  cpuKernelImpl,
  dtype
}: BinaryKernelFuncConfig): KernelFunc {
  return ({inputs, backend}) => {
    const {a, b} = inputs as BinaryInputs;
    const webglBackend = backend as MathBackendWebGL;

    if (supportsComplex && a.dtype === 'complex64') {
      const aData = webglBackend.texData.get(a.dataId);
      const bData = webglBackend.texData.get(b.dataId);

      const [real, imag] = [
        [aData.complexTensorInfos.real, bData.complexTensorInfos.real],
        [aData.complexTensorInfos.imag, bData.complexTensorInfos.imag]
      ].map(complexParts => {
        const [aPart, bPart] = complexParts;

        const aHandle = {
          dataId: aPart.dataId,
          dtype: aPart.dtype,
          shape: a.shape
        };
        const bHandle = {
          dataId: bPart.dataId,
          dtype: bPart.dtype,
          shape: b.shape
        };

        const program = new BinaryOpProgram(opSnippet, a.shape, b.shape);
        return webglBackend.runWebGLProgram(
            program, [aHandle, bHandle], upcastType(aPart.dtype, bPart.dtype));
      });

      const complexOutput =
          complex({inputs: {real, imag}, backend: webglBackend});

      webglBackend.disposeIntermediateTensorInfo(real);
      webglBackend.disposeIntermediateTensorInfo(imag);

      // TODO(annxingyuan): Implement CPU forwarding for complex inputs.

      return complexOutput;
    }

    const $dtype = dtype || upcastType(a.dtype, b.dtype);
    if (webglBackend.shouldExecuteOnCPU([a, b]) && cpuKernelImpl != null) {
      const aData = webglBackend.texData.get(a.dataId);
      const bData = webglBackend.texData.get(b.dataId);
      const [outValues, outShape] = cpuKernelImpl(
          a.shape, b.shape, aData.values as TypedArray,
          bData.values as TypedArray, $dtype);

      const out = webglBackend.makeTensorInfo(outShape, $dtype);
      const outData = webglBackend.texData.get(out.dataId);
      outData.values = outValues;
      return out;
    }

    const shouldUsePackedProgram =
        env().getBool('WEBGL_PACK_BINARY_OPERATIONS') &&
        packedOpSnippet != null;
    let program: BinaryOpProgram|BinaryOpPackedProgram;
    if (shouldUsePackedProgram) {
      program = new BinaryOpPackedProgram(
          packedOpSnippet, a.shape, b.shape, checkOutOfBounds);
    } else {
      program = new BinaryOpProgram(opSnippet, a.shape, b.shape);
    }

    return webglBackend.runWebGLProgram(program, [a, b], $dtype);
  };
}
