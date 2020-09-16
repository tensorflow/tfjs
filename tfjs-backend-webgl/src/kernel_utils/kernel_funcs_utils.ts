import {BinaryInputs, DataType, env, KernelFunc, UnaryInputs} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';
import {UnaryOpProgram} from '../unaryop_gpu';

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
export function binaryKernelFunc(
    opSnippet: string, packedOpSnippet: string,
    checkOutOfBoundsForPackedProgram?: boolean, dtype?: DataType): KernelFunc {
  // TODO(jingjin): handle complex64.

  return ({inputs, backend}) => {
    const {a, b} = inputs as BinaryInputs;
    const webglBackend = backend as MathBackendWebGL;
    const program = env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(
            packedOpSnippet, a.shape, b.shape,
            !!checkOutOfBoundsForPackedProgram) :
        new BinaryOpProgram(opSnippet, a.shape, b.shape);
    const $dtype = dtype || a.dtype;
    const output = webglBackend.runWebGLProgram(program, [a, b], $dtype);
    return output;
  };
}
